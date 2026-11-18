"""Withdrawal estimates for a bright smooth nail; SI units, no simulator imports."""
from dataclasses import dataclass


NAIL_LENGTH = .0381
NAIL_DIAMETER = .0025146  # .099 in, 12-1/2 gauge common nail.
NAIL_HEAD_DIAMETER = .00635
NAIL_HEAD_THICKNESS = .0010  # Photo estimate; check the purchased batch.
INITIAL_EMBEDMENT = .020
SPECIFIC_GRAVITY = .32  # Western red cedar reference value, not a batch measurement.
ELASTIC_STIFFNESS = 1_000_000.  # N/m; numerical compliance, not measured wood stiffness.


def withdrawal_force(embedment_m, diameter_m=NAIL_DIAMETER, specific_gravity=SPECIFIC_GRAVITY):
    """USDA Wood Handbook eq. 8-1 average peak load, N.

    The handbook uses diameter and penetration in mm. This is an estimate for
    smooth bright nails in side grain, not an allowable structural design load.
    """
    if diameter_m <= 0 or specific_gravity <= 0:
        raise ValueError('Positive nail diameter and wood specific gravity required')
    return 54.12 * specific_gravity**2.5 * (diameter_m*1000) * (max(0.,embedment_m)*1000)


@dataclass
class Withdrawal:
    """Elastoplastic reference for an implicit force-limited joint drive.

    Only contact forces move the nail. The model changes the spring's plastic
    rest position after yield; it never writes a body pose or velocity.
    """
    plastic_position: float = 0.
    released: bool = False

    def update(self, extraction_m):
        if extraction_m >= INITIAL_EMBEDMENT+.001:
            self.released = True
        resistance = 0. if self.released else withdrawal_force(INITIAL_EMBEDMENT-extraction_m)
        # Monotonic plastic withdrawal with a small elastic unloading excursion.
        self.plastic_position = max(self.plastic_position,
                                    extraction_m-resistance/ELASTIC_STIFFNESS)
        return self.plastic_position, resistance
