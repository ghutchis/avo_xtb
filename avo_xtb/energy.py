# SPDX-FileCopyrightText: 2024 Matthew J. Milner <matterhorn103@proton.me>
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import logging
import sys

from support import easyxtb


logger = logging.getLogger(__name__)


def energy(avo_input: dict) -> dict:
    # Extract the coords
    geom = easyxtb.Geometry.from_cjson(avo_input["cjson"])

    # Run calculation; returns energy as float in hartree
    logger.debug("avo_xtb is requesting a single point energy calculation")
    calc = easyxtb.Calculation.sp(
        geom,
        options=easyxtb.config["xtb_opts"],
    )
    calc.run()
    
    # If an energy couldn't be parsed, will return None, so have to allow for that
    # Seems like a reasonable placeholder that should be obviously incorrect to anyone
    energy_hartree = 0.0 if calc.energy is None else calc.energy
        
    # Convert energy to eV for Avogadro, other units for users
    energies = easyxtb.convert.convert_energy(energy_hartree, "hartree")
    # Format everything appropriately for Avogadro
    # Start by passing back the original cjson, then add changes
    output = {"cjson": avo_input["cjson"].copy()}
    # Currently Avogadro ignores the energy result
    output["message"] = (
        f"Energy from GFN{easyxtb.config['method']}-xTB:\n"
        + f"{str(round(energy_hartree, 7))} hartree\n"
        + f"{str(round(energies['eV'], 7))} eV\n"
        + f"{str(round(energies['kJ'], 7))} kJ/mol\n"
        + f"{str(round(energies['kcal'], 7))} kcal/mol\n"
    )
    output["cjson"]["properties"]["totalEnergy"] = round(energies["eV"], 7)
    # Partial charges if present
    if hasattr(calc, "partial_charges"):
        output["cjson"]["atoms"]["partialCharges"] = calc.partial_charges

    # Save result
    with open(easyxtb.TEMP_DIR / "result.cjson", "w", encoding="utf-8") as save_file:
        json.dump(output["cjson"], save_file, indent=2)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--print-options", action="store_true")
    parser.add_argument("--run-command", action="store_true")
    parser.add_argument("--display-name", action="store_true")
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--menu-path", action="store_true")
    args = parser.parse_args()

    # Disable if xtb missing
    if easyxtb.XTB.path is None:
        quit()

    if args.print_options:
        options = {"inputMoleculeFormat": "xyz"}
        print(json.dumps(options))
    
    if args.display_name:
        print("Energy")
    
    if args.menu_path:
        print("Extensions|Semi-Empirical QM (xTB){890}")

    if args.run_command:
        # Read input from Avogadro
        avo_input = json.loads(sys.stdin.read())
        output = energy(avo_input)
        # Pass back to Avogadro
        print(json.dumps(output))
        logger.debug(f"The following dictionary was passed back to Avogadro: {output}")
