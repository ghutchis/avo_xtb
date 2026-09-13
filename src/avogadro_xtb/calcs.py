# SPDX-FileCopyrightText: 2026 Matthew Milner <matterhorn103@proton.me>
# SPDX-License-Identifier: BSD-3-Clause

import logging

import easyxtb

from .progress import reporter_for, streaming_kwargs


logger = logging.getLogger(__name__)

# How many times to let xtb restart a Smart Opt from a distorted geometry before
# giving up. xtb normally needs at most one or two; this only stops a pathological
# case from leaving the user in front of a dialog that never finishes.
MAX_OPT_RESTARTS = 10

# Piggyback the easyxtb config and add some extra plugin-specific things
plugin_defaults = {
    "energy_units": "kJ/mol",
    "xtb_opts": {},
    "crest_opts": {},
}
for k, v in plugin_defaults.items():
    if k not in easyxtb.config:
        easyxtb.config[k] = v


def sp(avo_input: dict) -> dict:
    cjson = avo_input["cjson"]
    geom = easyxtb.Geometry.from_cjson(cjson)

    # Run calculation; returns energy as float in hartree
    logger.debug("The plugin is requesting a single point energy calculation")
    calc = easyxtb.Calculation.sp(
        geom,
        options=easyxtb.config["xtb_opts"],
    )
    reporter = reporter_for("Calculating energy…")
    calc.run(**streaming_kwargs(calc.run, reporter))

    # If an energy couldn't be parsed, will return None, so have to allow for that
    # Seems like a reasonable placeholder that should be obviously incorrect to anyone
    energy_hartree = 0.0 if calc.energy is None else calc.energy

    # Convert energy to eV for Avogadro, other units for users
    energies = easyxtb.convert.convert_energy(energy_hartree, "hartree")

    # Add changes to cjson
    cjson["properties"]["totalEnergy"] = round(energies["eV"], 7)
    # Partial charges if present
    if hasattr(calc, "partial_charges"):
        cjson["atoms"]["partialCharges"] = calc.partial_charges

    # Format output appropriately for Avogadro
    output = {"cjson": cjson}
    # Currently Avogadro ignores the energy result so tell the user via a message
    output["message"] = (
        f"Energy from GFN{easyxtb.config['method']}-xTB:\n"
        + f"{str(round(energy_hartree, 7))} hartree\n"
        + f"{str(round(energies['eV'], 7))} eV\n"
        + f"{str(round(energies['kJ'], 7))} kJ/mol\n"
        + f"{str(round(energies['kcal'], 7))} kcal/mol\n"
    )

    return output


def cleanup_after_opt(cjson: dict) -> dict:
    """Returns a cjson dict minus any data that is no longer meaningful after
    a geometry change."""

    cleaned = cjson

    # Frequencies and orbitals
    for field in ["vibrations", "basisSet", "orbitals", "cube"]:
        if field in cleaned:
            del cleaned[field]
    # Atomic charges
    if "formalCharges" in cleaned["atoms"]:
        del cleaned["atoms"]["formalCharges"]
    if "partialCharges" in cleaned["atoms"]:
        del cleaned["atoms"]["partialCharges"]

    return cleaned


def apply_new_geometry(
    cjson: dict,
    geometry: easyxtb.Geometry,
    energy: float | None = None,
    partial_charges: dict | None = None,
) -> dict:
    """Replace the coordinates in a cjson dict with an optimized geometry, along with
    whatever else the calculation produced, and drop anything the geometry change has
    invalidated."""

    # Remove anything that is now unphysical after the optimization
    cjson = cleanup_after_opt(cjson)

    cjson["atoms"]["coords"] = geometry.to_cjson()["atoms"]["coords"]
    if energy is not None:
        energies = easyxtb.convert.convert_energy(energy, "hartree")
        cjson["properties"]["totalEnergy"] = round(energies["eV"], 7)
    else:
        # No energy could be parsed for the new geometry. The one already in the
        # cjson belongs to the geometry that has just been replaced, so keeping
        # it would report an energy that does not match the coordinates.
        cjson["properties"].pop("totalEnergy", None)
    if partial_charges is not None:
        cjson["atoms"]["partialCharges"] = partial_charges

    return cjson


def opt(avo_input: dict) -> dict:
    cjson = avo_input["cjson"]
    geom = easyxtb.Geometry.from_cjson(cjson)

    # Run calculation
    logger.debug("The plugin is requesting a geometry optimization")
    calc = easyxtb.Calculation.opt(
        geom,
        options=easyxtb.config["xtb_opts"],
    )
    reporter = reporter_for("Optimizing geometry…")
    calc.run(**streaming_kwargs(calc.run, reporter))

    # Check for convergence
    # TODO
    # Will need to look for "FAILED TO CONVERGE"

    cjson = apply_new_geometry(
        cjson,
        calc.output_geometry,
        energy=calc.energy,
        partial_charges=getattr(calc, "partial_charges", None),
    )

    # Format output appropriately for Avogadro
    output = {"moleculeFormat": "cjson", "cjson": cjson}

    return output


def smartopt(avo_input: dict) -> dict:
    """Optimize the geometry, restarting until the result is a genuine minimum.

    xtb reports imaginary frequencies by writing out a geometry distorted away from
    the stationary point and recommending a restart from it. Doing only one round, as
    this command used to, therefore hands back whatever stationary point was reached
    first, which for a symmetric starting geometry is often a transition state.
    """

    cjson = avo_input["cjson"]
    geom = easyxtb.Geometry.from_cjson(cjson)

    logger.debug("The plugin is requesting a restarting geometry optimization")
    reporter = reporter_for("Optimizing geometry…")

    # easyxtb only gained a version of this that hands back the whole Calculation in
    # 0.11; with an older one the restarting is still done, but the energy and partial
    # charges have to be recovered with a single point afterwards and the vibrational
    # frequencies are not available at all.
    smartopt_calculation = getattr(easyxtb.calculate, "smartopt_calculation", None)

    if smartopt_calculation is not None:
        calc = smartopt_calculation(
            geom,
            options=easyxtb.config["xtb_opts"],
            # Nothing guarantees that xtb ever reaches a minimum, and this is running
            # behind a dialog that a user is waiting on, so it cannot loop forever.
            max_restarts=MAX_OPT_RESTARTS,
            **streaming_kwargs(smartopt_calculation, reporter),
        )
        frequencies = getattr(calc, "frequencies", None)
        cjson = apply_new_geometry(
            cjson,
            calc.output_geometry,
            energy=calc.energy,
            partial_charges=getattr(calc, "partial_charges", None),
        )
    else:
        logger.debug("Installed easyxtb cannot return the Calculation, using fallback")
        optimized = easyxtb.calculate.smartopt(
            geom,
            options=easyxtb.config["xtb_opts"],
            **streaming_kwargs(easyxtb.calculate.smartopt, reporter),
        )
        frequencies = None
        if reporter is not None:
            reporter.report("Calculating final energy…")
        final = easyxtb.Calculation.sp(
            optimized,
            options=easyxtb.config["xtb_opts"],
        )
        final.run(**streaming_kwargs(final.run, reporter))
        cjson = apply_new_geometry(
            cjson,
            optimized,
            energy=final.energy,
            partial_charges=getattr(final, "partial_charges", None),
        )

    output = {"moleculeFormat": "cjson", "cjson": cjson}

    # The frequencies of the final geometry come for free with the runtype this uses,
    # so hand them over rather than making the user pay for them a second time.
    if frequencies:
        cjson["vibrations"] = easyxtb.convert.freq_to_cjson(frequencies)["vibrations"]
        # Reaching a minimum is the whole point of this command, so if one was not
        # found the user needs to be told rather than left to notice.
        if frequencies[0]["frequency"] < 0:
            output["message"] = (
                "A minimum could not be found!\n"
                + f"There is still an imaginary frequency after {MAX_OPT_RESTARTS} "
                + "restarts of the optimization.\n"
                + "The geometry may be a transition state.\n"
            )

    return output


def freq(avo_input: dict) -> dict:
    cjson = avo_input["cjson"]
    geom = easyxtb.Geometry.from_cjson(cjson)

    # Run calculation; returns set of frequency data
    logger.debug("The plugin is requesting a frequency calculation")
    reporter = reporter_for("Calculating vibrational frequencies…")
    freqs = easyxtb.calculate.frequencies(
        geom,
        options=easyxtb.config["xtb_opts"],
        **streaming_kwargs(easyxtb.calculate.frequencies, reporter),
    )

    # Format the frequencies in the appropriate way for CJSON
    freq_cjson = easyxtb.convert.freq_to_cjson(freqs)

    # Add frequency data to the CJSON
    cjson["vibrations"] = freq_cjson["vibrations"]

    # Format output appropriately for Avogadro
    output = {"moleculeFormat": "cjson", "cjson": cjson}

    # Inform user if there are negative frequencies
    if freqs[0]["frequency"] < 0:
        output["message"] = (
            "At least one negative frequency found!\n"
            + "This is not a minimum on the potential energy surface.\n"
            + "You should reoptimize the geometry.\n"
            + "This can be avoided in future by using the Smart Opt method."
        )

    return output


def orbitals(avo_input: dict) -> dict:
    cjson = avo_input["cjson"]
    geom = easyxtb.Geometry.from_cjson(cjson)

    # Run calculation; returns Molden output file as string
    logger.debug("The plugin is requesting a molecular orbitals calculation")
    reporter = reporter_for("Calculating molecular orbitals…")
    molden_string = easyxtb.calculate.orbitals(
        geom,
        options=easyxtb.config["xtb_opts"],
        **streaming_kwargs(easyxtb.calculate.orbitals, reporter),
    )

    # Format everything appropriately for Avogadro
    # Just pass orbitals file with instruction to read only properties
    output = {
        "readProperties": True,
        "moleculeFormat": "molden",
        "molden": molden_string,
        "cjson": cjson,
    }
    # As it stands, this means any other properties will be wiped
    # If there were e.g. frequencies in the original cjson, notify the user
    if "vibrations" in cjson:
        output["message"] = (
            "Calculation complete!\n"
            "The vibrational frequencies may have been lost in this process.\n"
            "Please recalculate them if they are missing and still desired.\n"
        )
    else:
        output["message"] = "Calculation complete!"

    # Save orbitals file as well
    with open(easyxtb.TEMP_DIR / "result.molden", "w", encoding="utf-8") as f:
        f.write(molden_string)

    return output
