# SPDX-FileCopyrightText: 2026 Matthew Milner <matterhorn103@proton.me>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the parsing of `xtb` output into Avogadro progress reports.

Every line of `xtb` output used here is copied verbatim from a real run of
`xtb 6.7`, including its leading whitespace, so that the tests fail if the
patterns are tightened to something the program does not actually print.
"""

import json

import pytest

from avogadro_xtb import progress
from avogadro_xtb.command import PROGRESS_ENVIRONMENT_VARIABLE

# Verbatim excerpt of an `--ohess` run: the optimizer banner, two optimization
# cycles, convergence, and the stages that follow the optimization.
OHESS_OUTPUT = """\
      -----------------------------------------------------------
     |                   =====================                   |
     |                        A N C O P T                        |
     |                   =====================                   |
      -----------------------------------------------------------

          ...................................................
          :                      SETUP                      :
          :.................................................:
          :   optimization level            normal          :
          :   max. optcycles                   200          :
          :.................................................:

........................................................................
.............................. CYCLE    1 ..............................
........................................................................

 iter      E             dE          RMSdq      gap      omega  full diag
   1    -13.4119708 -0.134120E+02  0.111E-04    1.88       0.0  T
     SCC iter.                  ...        0 min,  0.002 sec
     gradient                   ...        0 min,  0.001 sec
 * total energy  :   -13.2901472 Eh     change       -0.5965717E-10 Eh
   gradient norm :     0.2711023 Eh/α   predicted     0.0000000E+00 (-100.00%)
   displ. norm   :     0.6606687 α      lambda       -0.1446287E+00

........................................................................
.............................. CYCLE    2 ..............................
........................................................................
 * total energy  :   -13.4267662 Eh     change       -0.1366190E+00 Eh
   gradient norm :     0.2234698 Eh/α   predicted    -0.1038783E+00 ( -23.96%)

   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 2 ITERATIONS ***

           -------------------------------------------------
          |                Final Singlepoint                |
           -------------------------------------------------

           -------------------------------------------------
          |                Numerical Hessian                |
           -------------------------------------------------

           -------------------------------------------------
          |               Frequency Printout                |
           -------------------------------------------------

           -------------------------------------------------
          |             Thermodynamic Functions             |
           -------------------------------------------------

          | TOTAL ENERGY              -13.534167459894 Eh   |
          | GRADIENT NORM               0.000605352964 Eh/α |
"""


@pytest.fixture
def progress_enabled(monkeypatch):
    """Pretend Avogadro has asked for progress reports."""
    monkeypatch.setenv(PROGRESS_ENVIRONMENT_VARIABLE, "1")


def replay(output: str, capsys, initial_message=None) -> list[dict]:
    """Feed `output` through a reporter and return the payloads it emitted."""
    reporter = progress.XtbProgressReporter(initial_message)
    for line in output.splitlines():
        reporter(line)
    captured = capsys.readouterr().out
    payloads = []
    for line in captured.splitlines():
        envelope = json.loads(line)
        # Avogadro only strips a line from the output when it is an object with
        # exactly one key, "avogadro", so anything else would corrupt the result.
        assert list(envelope.keys()) == ["avogadro"]
        payloads.append(envelope["avogadro"])
    return payloads


def messages(output: str, capsys, initial_message=None) -> list[str]:
    return [p["message"] for p in replay(output, capsys, initial_message)]


def test_reports_nothing_without_the_environment_variable(capsys):
    """Avogadro 2.0.0 reads all of stdout as one JSON document, so a script must
    stay silent unless the running version has asked for progress reports."""
    reporter = progress.XtbProgressReporter("Optimizing geometry…")
    for line in OHESS_OUTPUT.splitlines():
        reporter(line)
    assert capsys.readouterr().out == ""


def test_reports_only_messages(progress_enabled, capsys):
    """`xtb` gives no trustworthy total for its optimizer, so the bar must be
    left indeterminate: a value or maximum would make Avogadro draw one."""
    payloads = replay(OHESS_OUTPUT, capsys)
    assert payloads
    for payload in payloads:
        assert set(payload) == {"message"}


def test_initial_message_is_reported_immediately(progress_enabled, capsys):
    reported = messages("", capsys, initial_message="Optimizing geometry…")
    assert reported == ["Optimizing geometry…"]


def test_reports_each_optimization_cycle(progress_enabled, capsys):
    reported = messages(OHESS_OUTPUT, capsys)
    assert "Optimizing geometry — cycle 1" in reported
    assert "Optimizing geometry — cycle 2" in reported


def test_reports_energy_and_gradient_of_each_cycle(progress_enabled, capsys):
    reported = messages(OHESS_OUTPUT, capsys)
    assert "Optimizing geometry — cycle 1, E = -13.290147 Eh, |g| = 0.271102" in reported
    assert "Optimizing geometry — cycle 2, E = -13.426766 Eh, |g| = 0.223470" in reported


def test_energy_alone_is_not_reported(progress_enabled, capsys):
    """The gradient norm follows the energy on the very next line, so reporting
    the energy on its own would only flicker a half-finished message past."""
    reported = messages(OHESS_OUTPUT, capsys)
    assert "Optimizing geometry — cycle 1, E = -13.290147 Eh" not in reported


def test_reports_convergence(progress_enabled, capsys):
    reported = messages(OHESS_OUTPUT, capsys)
    assert "Geometry converged after 2 cycles" in reported


def test_reports_the_stages_after_the_optimization(progress_enabled, capsys):
    """The extra stages of a Smart Opt are what distinguish it from a plain
    optimization, so each one should say what it is doing."""
    reported = messages(OHESS_OUTPUT, capsys)
    assert "Calculating vibrational frequencies…" in reported
    assert "Analysing vibrational frequencies…" in reported
    assert "Calculating thermodynamic functions…" in reported


def test_later_stages_are_reported_in_order(progress_enabled, capsys):
    """A stale cycle message must not be left on screen once the optimizer has
    handed over to the Hessian."""
    reported = messages(OHESS_OUTPUT, capsys)
    assert reported.index("Geometry converged after 2 cycles") < reported.index(
        "Calculating vibrational frequencies…"
    )
    assert reported[-1] == "Calculating thermodynamic functions…"


def test_summary_block_is_not_mistaken_for_a_stage(progress_enabled, capsys):
    """The final summary prints values between vertical bars in the same shape
    as a section banner."""
    reported = messages(OHESS_OUTPUT, capsys)
    assert not any("TOTAL ENERGY" in message for message in reported)


def test_repeated_messages_are_not_resent(progress_enabled, capsys):
    """The optimizer's banner spans two lines that both carry its title."""
    reported = messages(OHESS_OUTPUT, capsys)
    assert reported.count("Optimizing geometry…") == 1


def test_reports_failure_to_converge(progress_enabled, capsys):
    reported = messages(
        "   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 200 ITERATIONS ***",
        capsys,
    )
    assert reported == ["Geometry optimization did not converge"]


def test_singular_cycle_count(progress_enabled, capsys):
    reported = messages("   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 1 ITERATIONS ***", capsys)
    assert reported == ["Geometry converged after 1 cycle"]


# What a Smart Opt restart looks like: xtb returns to the optimizer with a
# geometry distorted away from the stationary point it just found.
RESTART_OUTPUT = """\
      -----------------------------------------------------------
     |                        A N C O P T                        |
      -----------------------------------------------------------
.............................. CYCLE    1 ..............................
 * total energy  :   -13.5000000 Eh     change       -0.1000000E+00 Eh
   gradient norm :     0.0100000 Eh/α   predicted    -0.1000000E+00 ( -10.00%)
"""


def test_restarted_optimization_reports_cycles_again(progress_enabled, capsys):
    """The reporter has to leave the frequency stage and go back to counting
    cycles when the optimizer starts over."""
    reported = messages(OHESS_OUTPUT + RESTART_OUTPUT, capsys)
    assert reported[-1] == (
        "Optimizing geometry (restart 1) — cycle 1, E = -13.500000 Eh, |g| = 0.010000"
    )


def test_restarts_are_counted(progress_enabled, capsys):
    """A user watching a Smart Opt should be able to tell a restart from the
    first attempt, since a restart means the first result was not a minimum."""
    reported = messages(OHESS_OUTPUT + RESTART_OUTPUT * 2, capsys)
    assert "Optimizing geometry (restart 1)…" in reported
    assert "Optimizing geometry (restart 2)…" in reported
    assert "Optimizing geometry (restart 3)…" not in reported


def test_the_first_optimization_is_not_called_a_restart(progress_enabled, capsys):
    """The optimizer's banner spans two lines carrying the same title, so a
    naive count of it would report a restart that never happened."""
    reported = messages(OHESS_OUTPUT, capsys)
    assert not any("restart" in message for message in reported)


def test_unrecognized_output_is_ignored(progress_enabled, capsys):
    """A future `xtb` that reformats its output should cost the progress
    display, not the calculation."""
    reported = messages("something entirely unexpected\n\n   |  ??  |\n", capsys)
    assert reported == []


def test_streaming_support_is_detected(progress_enabled):
    def with_hook(line, on_output_line=None):
        pass

    def without_hook(line):
        pass

    assert progress.supports_streaming(with_hook)
    assert not progress.supports_streaming(without_hook)


def test_streaming_kwargs_omits_the_hook_when_unsupported(progress_enabled):
    """The plugin has to keep working against a released `easyxtb` that does not
    offer the callback yet."""

    def with_hook(on_output_line=None):
        pass

    def without_hook():
        pass

    reporter = progress.XtbProgressReporter()
    assert progress.streaming_kwargs(with_hook, reporter) == {"on_output_line": reporter}
    assert progress.streaming_kwargs(without_hook, reporter) == {}


def test_streaming_kwargs_omits_the_hook_without_a_reporter():
    def with_hook(on_output_line=None):
        pass

    assert progress.streaming_kwargs(with_hook, None) == {}


def test_no_reporter_when_avogadro_does_not_support_progress(monkeypatch):
    monkeypatch.delenv(PROGRESS_ENVIRONMENT_VARIABLE, raising=False)
    assert progress.reporter_for("Optimizing geometry…") is None


def test_reporter_when_avogadro_supports_progress(progress_enabled):
    assert progress.reporter_for("Optimizing geometry…") is not None
