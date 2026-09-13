# SPDX-FileCopyrightText: 2026 Matthew Milner <matterhorn103@proton.me>
# SPDX-License-Identifier: BSD-3-Clause

"""Turn the running output of `xtb` into progress reports for Avogadro.

`xtb` flushes its standard output line by line, so the progress of a long
calculation can be followed while it is still running. `XtbProgressReporter`
consumes those lines and forwards a short status message to Avogadro whenever
something worth reporting happens.

Only messages are reported, never a progress bar range. `xtb` announces its
optimizer's limit as `max. optcycles`, but that is a worst-case cap (200 by
default) rather than an expectation: a typical optimization converges in a few
dozen cycles, so a bar scaled to it would crawl to a small fraction and then
jump straight to complete, and the estimated time remaining would be wrong by
an order of magnitude. The cycle count, energy and gradient norm are reported
as text instead, which is the information actually needed to follow a
convergence.
"""

import inspect
import logging
import re

from .command import progress_supported, report_progress

logger = logging.getLogger(__name__)


# Section banners, printed centred between vertical bars, mark xtb's transition
# from one stage of a calculation to the next. Keys are the banner titles with
# runs of whitespace collapsed to a single space; the spaced-out letters of
# xtb's display font are part of the title itself.
SECTION_MESSAGES = {
    "A N C O P T": "Optimizing geometry…",
    "Numerical Hessian": "Calculating vibrational frequencies…",
    "Final Singlepoint": "Final single point calculation…",
    "Frequency Printout": "Analysing vibrational frequencies…",
    "Thermodynamic Functions": "Calculating thermodynamic functions…",
}

# A banner line, e.g. "          |                Numerical Hessian                |".
# The summary block at the end of a run prints values in the same shape (for
# instance "| TOTAL ENERGY   -13.534167459894 Eh   |"), so a captured title is
# only used when it matches one of the sections above.
BANNER_RE = re.compile(r"^\|\s*(.+?)\s*\|$")

# "  .............................. CYCLE    1 .............................."
CYCLE_RE = re.compile(r"^\.{3,}\s*CYCLE\s+(\d+)\s*\.{3,}$")

# " * total energy  :   -13.2901472 Eh     change       -0.2635758E-10 Eh"
ENERGY_RE = re.compile(r"^\*\s*total energy\s*:\s*(-?\d+\.\d+)")

# "   gradient norm :     0.2711022 Eh/α   predicted     0.0000000E+00 (-100.00%)"
GRADIENT_RE = re.compile(r"^gradient norm\s*:\s*(-?\d+\.\d+)")

# "   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 20 ITERATIONS ***"
CONVERGED_RE = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATION")

# xtb gives up with "*** FAILED TO CONVERGE GEOMETRY OPTIMIZATION ... ***". The
# wording after that has varied between releases, so only the stable part is
# matched.
FAILED_RE = re.compile(r"FAILED TO CONVERGE GEOMETRY OPTIMIZATION")


class XtbProgressReporter:
    """Reports the progress of a running `xtb` calculation to Avogadro.

    An instance is callable and is intended to be handed straight to
    `easyxtb.Calculation.run()` as its `on_output_line` argument:

        calc.run(on_output_line=XtbProgressReporter("Optimizing geometry…"))

    Parsing is deliberately forgiving. Every pattern is anchored to wording that
    has been stable across `xtb` releases, and a line that matches nothing is
    simply ignored, so an unrecognized or reformatted output only costs the
    progress display, never the calculation.
    """

    def __init__(self, initial_message: str | None = None):
        """
        Parameters
        ----------
        initial_message
            Status to show before `xtb` has produced anything worth reporting,
            which covers the startup of the subprocess and its setup output.
        """
        self.cycle = None
        self.energy = None
        self.gradient = None
        # Set once the optimizer has finished, so that the cycle-by-cycle
        # reporting of a geometry optimization does not overwrite the message
        # for a later stage such as the numerical Hessian.
        self.optimization_done = False
        self._last_message = None

        if initial_message:
            self.report(initial_message)

    def report(self, message: str) -> None:
        """Send `message` to Avogadro, unless it is already being displayed."""
        if message == self._last_message:
            return
        self._last_message = message
        report_progress(message=message)

    def optimization_message(self) -> str:
        """Describe the state of the geometry optimization in one line."""
        message = f"Optimizing geometry — cycle {self.cycle}"
        # The energy and gradient norm of a cycle are only printed once its
        # single point is finished, so they are absent while the first cycle is
        # still being computed.
        if self.energy is not None:
            message += f", E = {self.energy:.6f} Eh"
        if self.gradient is not None:
            message += f", |g| = {self.gradient:.6f}"
        return message

    def __call__(self, line: str) -> None:
        """Consume one line of `xtb` output and report any progress it implies."""
        line = line.strip()

        cycle_match = CYCLE_RE.match(line)
        if cycle_match:
            self.cycle = int(cycle_match.group(1))
            # Belongs to a new cycle, so the values carried over from the
            # previous one are stale.
            self.energy = None
            self.gradient = None
            self.report(self.optimization_message())
            return

        if self.cycle is not None and not self.optimization_done:
            energy_match = ENERGY_RE.match(line)
            if energy_match:
                # Recorded but not reported yet: xtb prints the gradient norm on
                # the very next line, so reporting here would only flicker a
                # half-finished message past the user.
                self.energy = float(energy_match.group(1))
                return

            gradient_match = GRADIENT_RE.match(line)
            if gradient_match:
                self.gradient = float(gradient_match.group(1))
                self.report(self.optimization_message())
                return

        converged_match = CONVERGED_RE.search(line)
        if converged_match:
            self.optimization_done = True
            cycles = int(converged_match.group(1))
            plural = "" if cycles == 1 else "s"
            self.report(f"Geometry converged after {cycles} cycle{plural}")
            return

        if FAILED_RE.search(line):
            self.optimization_done = True
            self.report("Geometry optimization did not converge")
            return

        banner_match = BANNER_RE.match(line)
        if banner_match:
            title = " ".join(banner_match.group(1).split())
            message = SECTION_MESSAGES.get(title)
            if message is not None:
                # A banner is drawn over several lines, and the optimizer's
                # spans two, so the same title can arrive more than once;
                # report() drops the repeat.
                if title == "A N C O P T":
                    # xtb restarts the optimizer from a distorted geometry when
                    # it finds an imaginary frequency, so reaching this banner
                    # again means a fresh optimization is starting.
                    self.optimization_done = False
                    self.cycle = None
                self.report(message)


def supports_streaming(func) -> bool:
    """Report whether `func` accepts an `on_output_line` callback.

    Streaming the output of `xtb` line by line needs a version of `easyxtb`
    that offers this hook. Older versions collect the output and hand it over
    only once the calculation has finished, which is too late to report
    anything, so the plugin checks rather than assumes and simply runs without
    progress reporting when the hook is missing.
    """
    try:
        return "on_output_line" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        # Not introspectable, so there is no way to tell; assume it is missing.
        logger.debug("Could not inspect %r for streaming support", func)
        return False


def reporter_for(initial_message: str | None = None) -> XtbProgressReporter | None:
    """Make a reporter for the calculation about to be run, or `None` if there
    is nothing to report to.

    Avogadro only reads the progress protocol when it has asked for it, and
    versions before 2.1 do not understand it at all, so on those there is no
    point in streaming the output of `xtb`.
    """
    if not progress_supported():
        return None
    return XtbProgressReporter(initial_message)


def streaming_kwargs(func, reporter: XtbProgressReporter | None) -> dict:
    """Build the keyword arguments needed to pass `reporter` to `func`.

    Returns an empty dict when there is no reporter or when `func` cannot take
    one, so the result can always be applied to the call:

        calc.run(**streaming_kwargs(calc.run, reporter))
    """
    if reporter is None or not supports_streaming(func):
        return {}
    return {"on_output_line": reporter}
