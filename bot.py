# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal, Union

import numpy as np

from lunarlander import Instructions


def time_to_target(x, v, a):
    """
    Calculate the time to reach a certain distance with a given velocity and acceleration.
    """
    return (-v + np.sqrt(v ** 2 + 2 * a * x)) / a


def critical_distance(v, a):
    """
    Calculate the distance at which the ship will stop with a given velocity and acceleration.
    """
    return v ** 2 / (2 * a)


def heading_to_target(x, v):
    ang = np.degrees(np.arctan2(-x, v))
    head = ang
    special = 70.5  # approx arccos(1/3)
    if head < -special:
        head = -special
    elif head > special:
        head = special
    return head


def rotate_instruction(current: float, target: float) -> Instructions:
    instruction = Instructions()
    if abs(current - target) < 2:
        pass
    elif current < target:
        instruction.left = True
    else:
        instruction.right = True
    return instruction


def find_landing_site(terrain: np.ndarray) -> Union[int, None]:
    # Find largest landing site
    n = len(terrain)
    # Find run starts
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(terrain[:-1], terrain[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # Find run lengths
    run_lengths = np.diff(np.append(run_starts, n))

    # Find all landing regions (where the size of the run is at least 25)
    landing_regions = np.where(run_lengths > 48)
    # We get points for picking a landing site close in size to the lander, so now choose the smallest one
    if len(landing_regions[0]) > 0:
        landing_sites = run_starts[landing_regions]
        landing_sizes = run_lengths[landing_regions]
        imin = np.argmin(landing_sizes)
        loc = int(landing_sites[imin] + (landing_sizes[imin] * 0.5))
        print("Found landing site at", loc)
        return loc


class Bot:
    """
    This is the lander-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        from simple_pid import PID
        self.team = "Hopper"  # This is your team name
        self.avatar = 4  # Optional attribute
        self.flag = "aq"  # Optional attribute
        self.initial_manoeuvre = True
        self.target_site = None
        self.target_height= None
        self.loiter_height = 900 # 1090+36
        self.height_pid = PID(0.1, 0.1, 0.1, setpoint=self.loiter_height, output_limits=(-1, 1))
        self.horizontal_pid = PID(0.1, 0.1, 0.1, setpoint=0, output_limits=(-1000, 1000))

        self.max_angle = 70
        self.max_horizontal_speed = np.sin(np.deg2rad(self.max_angle)) * 3

    def go_right(self, head, vx, control=0):
        if vx < self.max_horizontal_speed:
            angle = control if control > -self.max_angle else -self.max_angle
            instruction = rotate_instruction(current=head, target=angle)
            if abs(control) > 100:
                instruction.main = True
        else:
            # prepare to hold altitude
            instruction = rotate_instruction(current=head, target=0)
        return instruction

    def go_left(self, head, vx, control=0):
        if vx > -self.max_horizontal_speed:
            angle = control if control < self.max_angle else self.max_angle
            instruction = rotate_instruction(current=head, target=angle)
            if abs(control) > 100:
                instruction.main = True
        else:
            # prepare to hold altitude
            instruction = rotate_instruction(current=head, target=0)
        return instruction

    def go_stop(self, head, vx, control=0):
        instruction = rotate_instruction(current=head, target=self.max_angle if vx > 0 else -self.max_angle)
        instruction.main = True
        return instruction

    def control_horiozontal(self, head, vx, control):
        if control > 0:
            return self.go_right(head, vx, -control)
        elif control < 0:
            return self.go_left(head, vx, -control)
        else:
            return self.go_stop(head, vx, control)

    def run(
        self,
        t: float,
        dt: float,
        terrain: np.ndarray,
        players: dict,
        asteroids: list,
    ):
        instructions = Instructions()

        me = players[self.team]
        x, y = me.position
        vx, vy = me.velocity
        # convert the heading to +/-180 degrees
        head = (me.heading + 180) % 360 - 180

        # Perform an initial rotation to get the LEM pointing upwards
        if self.target_site is None:
            # self.target_site = 1000
            self.target_site = find_landing_site(terrain)
            if self.target_site is not None:
                self.target_height = terrain[self.target_site]

        if self.target_site is None:
            # if np.abs(vx) > 5:
            #     return self.go_stop(head, vx)
            instructions = rotate_instruction(current=head, target=0)
        else:
            g = 1.62
            on_ax = -3 * g * np.sin(np.deg2rad(head))
            on_ay = 3 * g * np.cos(np.deg2rad(head)) - g
            off_ax = 0
            off_ay = -g

            half = 860
            x_clamp = (self.target_site - x + half) % (2 * half) - half

            # on_ty = time_to_target(self.target_height - y, vy, on_ay)
            # on_tx = time_to_target(x_clamp, vx, on_ax)
            # off_ty = time_to_target(self.target_height - y, vy, off_ay)
            # off_tx = time_to_target(x_clamp, vx, off_ax)
            #
            # print(f'{on_ax=} {on_ay=} {on_ty=} {on_tx=}')
            # print(f'{off_ax=} {off_ay=} {off_ty=} {off_tx=}')
            # print(f'{x_clamp=} {self.target_site=}')
            # print(f'{head=} {x=} {y=} {vx=} {vy=}')

            # if not (off_tx > 0) or (off_ty < off_tx):
            #     self.height_pid.setpoint = y

            # horizontal_control = self.horizontal_pid(x_clamp)
            # print(f"{x_clamp=} {horizontal_control=}")
            instructions = self.control_horiozontal(head, vx, x_clamp)

            if abs(x_clamp) < 10:
                self.height_pid.setpoint = (y - self.target_height) * 0.5 + self.target_height


            # elif critical_distance(vy, off_ay) > abs(self.target_height - y):
            #     self.height_pid.setpoint = self.target_height
            # else:
            #     instructions.main = True
            #     return
            #
            # command = rotate(current=head, target=heading_to_target(x_clamp, vx))
            # if command == "left":
            #     instructions.left = True
            # elif command == "right":
            #     instructions.right = True

        height_control = self.height_pid(y)

        if height_control > 0:
            instructions.main = True

        return instructions
