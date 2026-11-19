"""The scripted demo for oil-squeaky-hinge, run by demokit.py through task.demo_script.

Closed loop: every stage reads where the can, its straw and its actuator are now, and the door's angle (poses and
visible joints, as a policy's perception could), never the scorer's record. Nothing is teleported or edited.

1. The right arm picks the can up round its body from its right-hand side (fingers level, pointing left, one pad on
   the can's front and one on its back), lifts it and turns it so the straw is level and points straight at the hinge.
2. It brings the straw's tip 12 mm in front of the barrel, aimed so the straight stream would land at the lowest
   gap. The left arm comes down from behind onto the red pad with the underside of its closed, level fingers, lying
   along the pad over the valve, and presses it slowly to just short of opening (most of the force) while the right
   arm re-settles the aim (both arms are soft, so the right arm lifts as the left pushes and servoes on where the
   stream would land). It stays on the pad for all four gaps: at each, once the aim has settled, it presses the last
   fraction of a millimetre for a burst of about 0.3 s, holding the valve open only while the stream lands on the
   gap, then lets back to just shut while the right arm steps the straw up to the next gap. A gap the demo is not
   sure it hit gets a second burst if the spray budget allows. Then the left arm lets the pad up slowly.
3. The right arm stands the can back where it was and lets go.
4. The left arm takes the knob, fingers either side of it and tipped 30 deg toward the door, and swings the door
   open about 60 deg and shut again, three times; lets go.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import scoring as S
from demokit import DT, level, tilted

MM = L.MM
AX, AY = L.axis_world()
BARREL_FRONT_X = AX - L.BARREL_R
TIP_STANDOFF = 12 * MM  # the straw's tip this far in front of the barrel while spraying
HOLD_Z = 0.095  # the right hand grips the can's body this high above its base (can frame): on the label
GRIP_FWD = 6 * MM  # ...with the TCP this far past the can's axis
PRESS = level(0.0)  # the pressing hand: fingers level, closed, pointing forward along the pad
PRESS_TCP_X = 3 * MM  # its TCP over the pad (can frame): the closed fingers' underside lies over the valve stem
PRESS_START = 0.035  # the TCP comes down to this high over the pad's top, then on down until the pad starts to move
TOUCHED = 0.08 * MM  # the fingers have met the pad when it has moved this far
FEED = 1.1  # the holding hand lifts this much for each metre the pressing hand pushes down (both arms are soft)
HOLD_V = 1.35 * MM  # between bursts the pad is held pressed this far: most of the force, the valve shut (1.5 mm)
VALVE_WANT = 1.58 * MM  # a burst presses it this far (it opens at 1.5 mm)
V_PER_H = 1.0 / 30.0  # roughly how far the valve goes per metre the soft pressing arm's command goes down (loop gain)
DOSE_WANT = 0.30  # s of open valve with the tip at the gap, per gap
SPRAY_BUDGET = 4.3  # s: the demo keeps its total spraying under this (the scorer's limit, 4 g, is about 4.9 s)
SWING_DEG = 60.0
PARK = {"right": (0.18, -0.22, L.TOP + 0.25), "left": (0.25, 0.14, L.TOP + 0.20)}


# -- reading the scene ------------------------------------------------------------------------------------------------
def valve(ctx) -> float:
    return float(ctx.joint("can", "valve"))


def jet(ctx):
    s = ctx.state("can")
    return S.straw_jet(s.rotation(), s.origin, valve(ctx))


def tip(ctx) -> np.ndarray:
    return jet(ctx)[0]


def can_point(ctx, local) -> np.ndarray:
    return np.asarray(ctx.state("can").to_world(np.asarray(local, float)), float)


def aim_error(ctx, goal) -> np.ndarray:
    """How far to move the can so the stream lands at the gap: x from the tip's stand-off, y and z from where the
    straight stream from the tip meets the plane of the barrel's front (it pitches as the press tilts the can)."""
    t, d = jet(ctx)
    goal = np.asarray(goal, float)
    land = t + d * ((BARREL_FRONT_X - t[0]) / max(d[0], 0.2))
    return np.array([goal[0] - t[0], goal[1] - land[1], goal[2] - land[2]])


def gap_goal(k: int) -> np.ndarray:
    return np.array([BARREL_FRONT_X - TIP_STANDOFF, AY, S.seam_heights_world()[k]])


def report(ctx, label: str) -> None:
    t, d = jet(ctx)
    s = ctx.state("can")
    print(f"[demo] {label}: tip {np.round(t, 4)} dir {np.round(d, 3)} can tilt {math.degrees(s.tilt()):.1f} deg "
          f"valve {valve(ctx) / MM:.2f} mm door {math.degrees(ctx.joint('cabinet', 'door')):.1f} deg", flush=True)


# -- the can ------------------------------------------------------------------------------------------------------------
def pick_can(ctx, arm):
    """Take the can from its right-hand side, fingers level and pointing left, one pad on its front (under the
    straw) and one on its back: a press on the pad then cannot pitch it in the grip."""
    grip = level(90.0)
    at = lambda: can_point(ctx, (0.0, GRIP_FWD, HOLD_Z))  # noqa: E731
    yield from arm.move(at() + np.array([0.0, -0.06, 0.03]), grip=grip, finger=-0.04695, t=2.5, label="beside the can")
    yield from arm.move(at() + np.array([0.0, -0.06, 0.0]), t=0.8, label="down beside the can")
    yield from arm.move(at(), t=1.5, label="round the can")
    yield from arm.servo(at(), timeout=1.0, label="round the can")
    yield from arm.close(2 * L.CAN_R, force=50.0, label="grip the can")
    yield from arm.move(arm.at() + np.array([0.0, 0.0, 0.10]), t=1.5, label="lift the can")


def square_straw(ctx, arm):
    """Turn the held can so the straw is level and points along +x (straight at the hinge)."""
    for _ in range(2):
        _, d = jet(ctx)
        yaw = math.degrees(math.atan2(d[1], d[0]))
        if abs(yaw) > 0.5:
            yield from arm.turn(-yaw, axis=(0.0, 0.0, 1.0), t=max(0.5, abs(yaw) / 30.0), label="aim the straw")
        _, d = jet(ctx)
        pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
        if abs(pitch) > 0.5:  # nose up is +pitch: turn about +y (which tips +x down) by +pitch
            yield from arm.turn(pitch, axis=(0.0, 1.0, 0.0), t=max(0.5, abs(pitch) / 30.0), label="level the straw")
        yield from arm.wait(0.3)
    report(ctx, "straw squared")


def bring_tip(ctx, arm, goal, t: float = 1.5, label: str = "tip"):
    """Move the can so the straw's tip stands off the barrel and its stream lands at ``goal``'s height."""
    yield from arm.move(arm.at() + aim_error(ctx, goal), t=t, label=label)
    yield from arm.servo(goal, tol=0.0008, timeout=2.0, read=lambda: np.asarray(goal, float) - aim_error(ctx, goal),
                         label=label)


def hold_tip(ctx, arm, sh: dict, gain: float = 0.10, step: float = 0.0012):
    """Keep the stream landing at ``sh['goal']`` (which the pressing hand steps from gap to gap) until ``sh['done']``:
    a slow servo on where the stream would land, smoothed so it does not chase the arms' wobble; the goal's own moves
    are fed straight through, and ``sh['lead']`` lifts the hand as the other one presses down (both arms are soft)."""
    c = arm.at() + arm.trim
    last = np.asarray(sh["goal"], float).copy()
    ef = aim_error(ctx, last)
    while not sh.get("done"):
        goal = np.asarray(sh["goal"], float)
        c = c + (goal - last)
        last = goal.copy()
        ef = ef + 0.35 * (aim_error(ctx, goal) - ef)
        c = c + np.clip(gain * ef, -step, step)
        arm._solve(c + np.array([0.0, 0.0, sh.get("lead", 0.0)]), arm.grip.rot)
        yield


def tip_error(ctx, goal) -> np.ndarray:
    """Where the stream would land, less where it should (and the tip's stand-off error in x)."""
    return -aim_error(ctx, goal)


def on_gap(e) -> bool:
    return abs(e[2]) <= 0.8 * MM and math.hypot(e[0], e[1]) <= 2.5 * MM


def on_target(e) -> bool:
    return abs(e[2]) <= 1.8 * MM and math.hypot(e[0], e[1]) <= 3.5 * MM


def press_point(ctx, h: float = 0.0) -> np.ndarray:
    """The pressing hand's TCP over the pad (the pad unpressed, on the can as it is now), ``h`` above its top."""
    return can_point(ctx, (PRESS_TCP_X, 0.0, L.ACT_TOP + h))


def presser(ctx, arm, sh: dict, gaps, dose_want: float = DOSE_WANT, budget: float = SPRAY_BUDGET,
            burst_open_max: float = 1.0, burst_s: float = 2.5):
    """The pressing hand: comes down from behind and above onto the red pad with the underside of its closed
    fingers, which lie along the pad over the valve, and stays on it for every gap. Between bursts it holds the pad
    pressed just short of opening (most of the force) while the other hand steps the straw from gap to gap; at a gap,
    once the aim has settled, it presses the last fraction of a millimetre for a short burst, only while the stream
    lands on the gap, then lets back to just shut. At the end it lets the pad up slowly, the other hand letting its
    lift go with it. Fills ``sh['log']`` and sets ``sh['done']``."""
    yield from arm.move(press_point(ctx, PRESS_START + 0.03) - np.array([0.03, 0.0, 0.0]), grip=PRESS, finger=0.0,
                        t=1.5, label="behind and over the pad")
    yield from arm.move(press_point(ctx, PRESS_START), t=0.8, label="over the pad")
    top = arm.at() + arm.trim - np.array([0.0, 0.0, PRESS_START])  # the command for the TCP at the pad's top
    st = {"h": PRESS_START, "vf": valve(ctx), "ef": tip_error(ctx, sh["goal"]), "open": 0.0, "k": 0, "lost": False,
          "h0": None}
    base = {"p": press_point(ctx)}

    def tick(want=None, rate=0.08, cap=0.0008, lift=0.0):
        """One control step: smooth the readings, move the hand toward pressing the valve ``want`` far (or by
        ``lift``), keep the other hand's lift matched, count open-valve time. Returns the raw valve reading."""
        nonlocal top
        v = valve(ctx)
        st["vf"] += 0.3 * (v - st["vf"])
        st["ef"] += 0.35 * (tip_error(ctx, sh["goal"]) - st["ef"])
        now = press_point(ctx)
        top = top + 0.08 * (now - base["p"])  # follow the can slowly, not its wobble
        base["p"] = base["p"] + 0.08 * (now - base["p"])
        if want is not None:
            st["h"] -= float(np.clip(rate * (want - st["vf"]) / V_PER_H, -cap, cap))
        st["h"] += lift
        if st["h0"] is not None:
            sh["lead"] = FEED * max(0.0, st["h0"] - st["h"])
        arm._solve(top + np.array([0.0, 0.0, st["h"]]), PRESS.rot)
        if v >= L.SPRAY_AT:
            st["open"] += DT
        if np.linalg.norm(st["ef"]) > 0.03 and not st["lost"]:
            st["lost"] = True
            print(f"[demo] press: the aim is {np.round(st['ef'] / MM, 1)} mm off: letting the pad up", flush=True)
        st["k"] += 1
        if st["k"] % 5 == 0:
            print(f"[demo]   press: hand {st['h'] * 1000:.0f} mm down, valve {v / MM:.2f} mm, aim err "
                  f"{np.round(st['ef'] / MM, 1)} mm, open {st['open']:.2f} s", flush=True)
        return v

    def shift(new):  # the other hand is moving the can by this much: move with it
        nonlocal top
        d = np.asarray(new, float) - np.asarray(sh["goal"], float)
        top = top + d
        base["p"] = base["p"] + d
        sh["goal"] = new

    for _ in range(int(3.0 / DT)):  # down until the fingers meet the pad (it starts to move)
        if tick(lift=-0.0010) >= TOUCHED:
            break
        yield
    st["h0"] = st["h"]
    ok = 0
    for _ in range(int(6.0 / DT)):  # push it to just short of opening, slowly, the other hand lifting with it
        tick(HOLD_V)
        yield
        ok = ok + 1 if abs(st["vf"] - HOLD_V) < 0.05 * MM else 0
        if ok >= 6 or st["lost"]:
            break
    sh["log"] = []
    prev = None
    for k in gaps:
        if st["lost"]:
            break
        goal = gap_goal(k)
        if prev is not None:  # step the straw up to the next gap, the pad still held just shut
            n = int(1.2 / DT)
            for i in range(1, n + 1):
                shift(prev + (goal - prev) * i / n)
                tick(HOLD_V)
                yield
        shift(goal)
        prev = goal
        dosed = 0.0
        for attempt in range(2):
            ok = 0
            for _ in range(int(4.0 / DT)):  # wait for the aim to settle on the gap
                tick(HOLD_V)
                yield
                ok = ok + 1 if on_gap(st["ef"]) and abs(st["vf"] - HOLD_V) < 0.1 * MM else 0
                if ok >= 8 or st["lost"]:
                    break
            opened = st["open"]
            for _ in range(int(burst_s / DT)):  # the burst: open only while the stream lands on the gap
                if st["lost"]:
                    break
                hit = on_target(st["ef"])
                v = tick(VALVE_WANT if hit else HOLD_V, rate=0.25, cap=0.0012)
                yield
                if v >= L.SPRAY_AT and hit:
                    dosed += DT
                if dosed >= dose_want or st["open"] - opened >= burst_open_max or st["open"] >= budget:
                    break
            for _ in range(int(1.0 / DT)):  # straight back to just shut
                tick(HOLD_V, rate=0.3, cap=0.0015)
                yield
                if st["vf"] < HOLD_V + 0.05 * MM:
                    break
            print(f"[demo] gap {k + 1}: burst {attempt + 1}, {dosed:.2f} s on the gap, valve open {st['open']:.2f} s "
                  f"in all", flush=True)
            if st["lost"] or dosed >= dose_want - 1e-9 or st["open"] + 0.6 * (len(gaps) - gaps.index(k)) > budget:
                break
        sh["log"].append((k, dosed))
    while st["h"] < st["h0"] + 0.008:  # let the pad up slowly: the other hand lets its lift go as the force goes
        tick(lift=0.0015 if st["vf"] > HOLD_V - 0.3 * MM else 0.0006)
        yield
    yield from arm.move(arm.at() + np.array([-0.02, 0.0, 0.04]), t=0.8, label="off the pad")
    sh["open"] = st["open"]
    sh["done"] = True
    print(f"[demo] pressing done: valve open {st['open']:.2f} s in all", flush=True)


def spray_gaps(ctx, r, l, gaps=None, **kw):
    gaps = list(range(S.N_SEAMS)) if gaps is None else list(gaps)
    yield from square_straw(ctx, r)
    first = gap_goal(gaps[0])
    yield from bring_tip(ctx, r, first - np.array([0.05, 0.0, 0.0]), t=2.5, label="straw toward the hinge")
    yield from square_straw(ctx, r)
    yield from bring_tip(ctx, r, first, t=1.0, label=f"tip at gap {gaps[0] + 1}")
    report(ctx, f"at gap {gaps[0] + 1}")
    sh = {"goal": first}
    yield from ctx.together(presser(ctx, l, sh, gaps, **kw), hold_tip(ctx, r, sh))
    report(ctx, f"sprayed (valve open {sh.get('open', 0.0):.2f} s)")
    yield from l.move(PARK["left"], t=1.5, label="left aside")
    # the straw back out from the hinge, then the can down where it was
    yield from r.move(r.at() + np.array([-0.06, 0.0, 0.0]), t=1.0, label="straw back")


def set_can_down(ctx, arm):
    s = ctx.state("can")
    held = arm.at() - np.asarray(s.origin, float)  # the TCP from the can's base, as held
    goal = np.array([L.CAN_START[0], L.CAN_START[1], L.TOP + 0.003])
    yield from arm.move(goal + held + np.array([0.0, 0.0, 0.04]), t=2.5, label="can over its spot")
    yield from arm.move(goal + held + np.array([0.0, 0.0, 0.01]), t=1.0, label="can down")
    yield from arm.servo(goal, tol=0.002, timeout=2.0, read=lambda: np.asarray(ctx.state("can").origin, float),
                         label="can down on the table")
    yield from arm.wait(0.3)
    yield from arm.open(label="let go of the can")
    yield from arm.move(arm.at() - 0.08 * arm.approach_axis(), t=1.0, label="back off the can")
    yield from arm.move(PARK[arm.side], t=1.5, label="right aside")


# -- the door ---------------------------------------------------------------------------------------------------------------
def knob_world(ctx) -> np.ndarray:
    s = ctx.state("cabinet")
    return np.asarray(s.link_to_world("door", (-L.NECK_L - L.KNOB_L / 2, L.KNOB_Y, L.KNOB_Z)), float)


def work_door(ctx, arm, swings: int = 3):
    """Take the knob with the fingers either side of it, tipped 30 deg toward the door, and swing the door about its
    pin, the grip turning with it, open about 60 deg and shut, ``swings`` times; let go."""
    grip = tilted(30.0, 0.0)
    app = arm.approach_axis(grip)
    at = knob_world(ctx) - 0.007 * app  # the pad tips 6 mm past the knob's centre line
    yield from arm.move(at - 0.07 * app, grip=grip, finger=-0.04695, t=2.5, label="in front of the knob")
    yield from arm.move(at, t=1.2, label="fingers either side of the knob")
    yield from arm.servo(at, timeout=1.0, label="fingers either side of the knob")
    yield from arm.close(L.KNOB_D, force=30.0, label="take the knob")
    centre = np.array([AX, AY, at[2]])
    for n in range(swings):
        yield from arm.arc(centre, (0.0, 0.0, 1.0), SWING_DEG, t=2.2, label=f"open {n + 1}")
        print(f"[demo] swing {n + 1}: door open {math.degrees(ctx.joint('cabinet', 'door')):.1f} deg", flush=True)
        yield from arm.arc(centre, (0.0, 0.0, 1.0), -SWING_DEG, t=2.2, label=f"shut {n + 1}")
        yield from arm.wait(0.2)
        print(f"[demo] swing {n + 1}: door back to {math.degrees(ctx.joint('cabinet', 'door')):.1f} deg", flush=True)
    yield from arm.open(label="let go of the knob")
    yield from arm.move(arm.at() - 0.07 * arm.approach_axis(), t=1.0, label="back off the knob")
    yield from arm.move(PARK[arm.side], t=1.5, label="left aside")


# -- the whole task ---------------------------------------------------------------------------------------------------------
def flood(ctx, r, l):
    """Cheat check: hold the valve open about 5 s at the lowest gap."""
    yield from spray_gaps(ctx, r, l, gaps=[0], dose_want=99.0, budget=99.0, burst_open_max=5.2, burst_s=8.0)


def demo_script(ctx):
    r, l = ctx.right, ctx.left
    report(ctx, "start")
    if ctx.mode == "swing":
        yield from work_door(ctx, l)
        return
    yield from pick_can(ctx, r)
    report(ctx, "picked")
    if ctx.mode == "aim":
        yield from square_straw(ctx, r)
        yield from bring_tip(ctx, r, gap_goal(0), t=2.5, label="tip at gap 1")
        report(ctx, "aimed")
        return
    if ctx.mode == "flood":
        yield from flood(ctx, r, l)
        return
    yield from spray_gaps(ctx, r, l)
    if ctx.mode == "spray":
        return
    yield from set_can_down(ctx, r)
    report(ctx, "can down")
    yield from work_door(ctx, l)
    report(ctx, "end")
