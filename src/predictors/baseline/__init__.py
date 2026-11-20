"""Mixins used by the baseline predictor.

No package-level imports: the web imports helpers such as ``race.result_processing``,
and an eager import here loaded every mixin, and through them FastF1, into the web.
Import each mixin from its own module.
"""
