"""Early MAVSDK-based IMU grid traversal demo for a Pixhawk 4 companion computer.

This script is intentionally conservative:
- It uses MAVSDK for high-level control of the Pixhawk 4.
- It uses IMU dead reckoning only as a short-duration estimate.
- It assumes the drone starts at a known corner with the correct initial heading.

Important:
- IMU-only position estimation drifts quickly and should not be treated as reliable
  long-term localisation.
- PX4 Offboard mode generally expects a valid position or pose source. This script
  is best treated as a controlled prototype scaffold, not flight-ready autonomy.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
except ImportError as exc:  # pragma: no cover - dependency is optional in this environment
    raise SystemExit(
        "MAVSDK is required for this demo. Install it with `pip install mavsdk`."
    ) from exc


GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class SearchArea:
    """Rectangular search area in the local planning frame."""

    width_m: float
    length_m: float
    row_spacing_m: float
    scan_altitude_m: float


@dataclass(frozen=True)
class Waypoint:
    """Planned waypoint in the local search frame."""

    x_m: float
    y_m: float
    z_m: float
    tolerance_m: float
    max_speed_m_s: float


@dataclass
class EstimatedState:
    """Current IMU-based estimate of the drone state."""

    position_m: np.ndarray
    velocity_m_s: np.ndarray
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    last_update_s: Optional[float] = None


@dataclass
class ImuSample:
    """One IMU sample expressed in the drone body frame."""

    accel_body_m_s2: np.ndarray
    gyro_body_rad_s: np.ndarray
    timestamp_s: float


def parse_args() -> argparse.Namespace:
    """Parse runtime options for the IMU grid traversal demo."""
    parser = argparse.ArgumentParser(description="MAVSDK IMU-based snake grid traversal demo")
    parser.add_argument("--system-address", default="serial:///dev/ttyAMA0:921600", help="MAVSDK system address")
    parser.add_argument("--width-m", type=float, required=True, help="Search area width from left to right")
    parser.add_argument("--length-m", type=float, required=True, help="Search area length from top to bottom")
    parser.add_argument("--row-spacing-m", type=float, required=True, help="Distance between scan rows")
    parser.add_argument("--scan-altitude-m", type=float, default=1.5, help="Takeoff and scan altitude")
    parser.add_argument("--segment-speed-m-s", type=float, default=0.3, help="Nominal body-frame segment speed")
    parser.add_argument("--waypoint-tolerance-m", type=float, default=0.2, help="Waypoint completion tolerance")
    parser.add_argument("--hold-time-s", type=float, default=1.0, help="Hold at each waypoint before continuing")
    parser.add_argument("--max-mission-time-s", type=float, default=180.0, help="Maximum allowed mission duration")
    parser.add_argument("--max-position-error-m", type=float, default=3.0, help="Abort if estimate magnitude exceeds this")
    parser.add_argument("--accel-lpf-alpha", type=float, default=0.25, help="Acceleration low-pass factor")
    parser.add_argument("--attitude-correction-gain", type=float, default=0.02, help="Roll/pitch accel correction gain")
    parser.add_argument("--battery-min-fraction", type=float, default=0.30, help="Minimum battery fraction before arming")
    return parser.parse_args()


def generate_snake_path(area: SearchArea, tolerance_m: float, speed_m_s: float) -> list[Waypoint]:
    """Generate a top-left-origin snake path over the rectangular search area."""
    waypoints: list[Waypoint] = [
        Waypoint(0.0, 0.0, area.scan_altitude_m, tolerance_m=tolerance_m, max_speed_m_s=speed_m_s)
    ]
    y_m = 0.0
    row_index = 0

    while True:
        x_target = area.width_m if row_index % 2 == 0 else 0.0
        waypoints.append(
            Waypoint(
                x_target,
                y_m,
                area.scan_altitude_m,
                tolerance_m=tolerance_m,
                max_speed_m_s=speed_m_s,
            )
        )

        if y_m >= area.length_m:
            break

        next_y_m = min(y_m + area.row_spacing_m, area.length_m)
        waypoints.append(
            Waypoint(
                x_target,
                next_y_m,
                area.scan_altitude_m,
                tolerance_m=tolerance_m,
                max_speed_m_s=min(speed_m_s, 0.25),
            )
        )

        if next_y_m >= area.length_m:
            row_index += 1
            y_m = next_y_m
            continue

        y_m = next_y_m
        row_index += 1

    return _dedupe_consecutive_waypoints(waypoints)


def _dedupe_consecutive_waypoints(waypoints: list[Waypoint]) -> list[Waypoint]:
    """Remove repeated consecutive points caused by edge clamping."""
    deduped: list[Waypoint] = []
    for waypoint in waypoints:
        if not deduped:
            deduped.append(waypoint)
            continue
        prev = deduped[-1]
        if (
            abs(prev.x_m - waypoint.x_m) < 1e-6
            and abs(prev.y_m - waypoint.y_m) < 1e-6
            and abs(prev.z_m - waypoint.z_m) < 1e-6
        ):
            continue
        deduped.append(waypoint)
    return deduped


class ImuDeadReckoner:
    """Very simple IMU-only dead reckoning estimator for short prototype runs."""

    def __init__(self, accel_lpf_alpha: float = 0.25, attitude_correction_gain: float = 0.02) -> None:
        self._accel_lpf_alpha = float(np.clip(accel_lpf_alpha, 0.0, 1.0))
        self._attitude_correction_gain = float(np.clip(attitude_correction_gain, 0.0, 0.2))
        self._accel_bias = np.zeros(3, dtype=np.float64)
        self._gyro_bias = np.zeros(3, dtype=np.float64)
        self._filtered_accel_body = np.zeros(3, dtype=np.float64)
        self._bias_ready = False
        self.state = EstimatedState(
            position_m=np.zeros(3, dtype=np.float64),
            velocity_m_s=np.zeros(3, dtype=np.float64),
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.0,
        )

    def set_stationary_bias(self, accel_samples: list[np.ndarray], gyro_samples: list[np.ndarray]) -> None:
        """Estimate biases while the drone is stationary before takeoff."""
        if not accel_samples or not gyro_samples:
            raise ValueError("Need stationary accelerometer and gyroscope samples for bias estimation")

        accel_mean = np.mean(np.array(accel_samples, dtype=np.float64), axis=0)
        gyro_mean = np.mean(np.array(gyro_samples, dtype=np.float64), axis=0)

        self._gyro_bias = gyro_mean
        self._accel_bias = accel_mean - np.array([0.0, 0.0, -GRAVITY_M_S2], dtype=np.float64)
        self._filtered_accel_body = accel_mean.copy()
        self._bias_ready = True

    def reset_horizontal_velocity(self) -> None:
        """Zero the horizontal velocity estimate during a deliberate hover pause."""
        self.state.velocity_m_s[0] = 0.0
        self.state.velocity_m_s[1] = 0.0

    def update(self, sample: ImuSample) -> EstimatedState:
        """Update the dead-reckoning state using one IMU sample."""
        if not self._bias_ready:
            raise RuntimeError("IMU bias must be estimated before calling update()")

        if self.state.last_update_s is None:
            self.state.last_update_s = sample.timestamp_s
            return self.state

        dt = sample.timestamp_s - self.state.last_update_s
        self.state.last_update_s = sample.timestamp_s
        if dt <= 0.0 or dt > 0.5:
            return self.state

        gyro_rad_s = sample.gyro_body_rad_s - self._gyro_bias
        accel_body_m_s2 = sample.accel_body_m_s2 - self._accel_bias

        alpha = self._accel_lpf_alpha
        self._filtered_accel_body = (1.0 - alpha) * self._filtered_accel_body + alpha * accel_body_m_s2

        self.state.roll_rad += float(gyro_rad_s[0] * dt)
        self.state.pitch_rad += float(gyro_rad_s[1] * dt)
        self.state.yaw_rad += float(gyro_rad_s[2] * dt)

        accel_norm = float(np.linalg.norm(self._filtered_accel_body))
        if accel_norm > 1e-6:
            accel_unit = self._filtered_accel_body / accel_norm
            measured_roll = math.atan2(accel_unit[1], accel_unit[2] if accel_unit[2] != 0.0 else 1e-6)
            measured_pitch = math.atan2(
                -accel_unit[0],
                math.sqrt(accel_unit[1] ** 2 + accel_unit[2] ** 2),
            )
            gain = self._attitude_correction_gain
            self.state.roll_rad = (1.0 - gain) * self.state.roll_rad + gain * measured_roll
            self.state.pitch_rad = (1.0 - gain) * self.state.pitch_rad + gain * measured_pitch

        rot_world_from_body = _rotation_matrix(
            self.state.roll_rad,
            self.state.pitch_rad,
            self.state.yaw_rad,
        )
        accel_world = rot_world_from_body @ self._filtered_accel_body
        accel_world[2] += GRAVITY_M_S2

        self.state.velocity_m_s += accel_world * dt
        self.state.position_m += self.state.velocity_m_s * dt
        return self.state


def _rotation_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Build a ZYX rotation matrix from body frame to world/search frame."""
    sr, cr = math.sin(roll_rad), math.cos(roll_rad)
    sp, cp = math.sin(pitch_rad), math.cos(pitch_rad)
    sy, cy = math.sin(yaw_rad), math.cos(yaw_rad)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


class MavsdkImuGridDemo:
    """Runs a snake-grid mission using MAVSDK offboard velocity commands."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._area = SearchArea(
            width_m=args.width_m,
            length_m=args.length_m,
            row_spacing_m=args.row_spacing_m,
            scan_altitude_m=args.scan_altitude_m,
        )
        self._waypoints = generate_snake_path(
            self._area,
            tolerance_m=args.waypoint_tolerance_m,
            speed_m_s=args.segment_speed_m_s,
        )
        self._drone = System()
        self._estimator = ImuDeadReckoner(
            accel_lpf_alpha=args.accel_lpf_alpha,
            attitude_correction_gain=args.attitude_correction_gain,
        )

        self._latest_imu: Optional[ImuSample] = None
        self._latest_battery_fraction: Optional[float] = None
        self._telemetry_tasks: list[asyncio.Task[None]] = []
        self._offboard_started = False

    async def connect_to_pixhawk(self) -> None:
        """Connect MAVSDK to the Pixhawk 4."""
        await self._drone.connect(system_address=self._args.system_address)
        print(f"Connecting to Pixhawk 4 via {self._args.system_address}")

        async for state in self._drone.core.connection_state():
            if state.is_connected:
                print("Connected to Pixhawk 4")
                break

        self._telemetry_tasks.append(asyncio.create_task(self._stream_imu()))
        self._telemetry_tasks.append(asyncio.create_task(self._stream_battery()))

    async def run_preflight_checks(self) -> None:
        """Run conservative preflight checks before arming."""
        if self._area.width_m <= 0.0 or self._area.length_m <= 0.0:
            raise ValueError("Search area width and length must be positive")
        if self._area.row_spacing_m <= 0.0:
            raise ValueError("Row spacing must be positive")
        if self._args.segment_speed_m_s <= 0.0:
            raise ValueError("Segment speed must be positive")
        if self._args.segment_speed_m_s > 0.6:
            raise ValueError("Segment speed is too high for this early prototype; keep it at or below 0.6 m/s")

        battery_wait_started = time.monotonic()
        while self._latest_battery_fraction is None and time.monotonic() - battery_wait_started < 5.0:
            await asyncio.sleep(0.1)
        if self._latest_battery_fraction is None:
            raise RuntimeError("Battery telemetry unavailable")
        if self._latest_battery_fraction < self._args.battery_min_fraction:
            raise RuntimeError(
                f"Battery too low for mission: {self._latest_battery_fraction:.2f} < {self._args.battery_min_fraction:.2f}"
            )

        print("Waiting for IMU telemetry")
        imu_wait_started = time.monotonic()
        while self._latest_imu is None and time.monotonic() - imu_wait_started < 5.0:
            await asyncio.sleep(0.05)
        if self._latest_imu is None:
            raise RuntimeError("IMU telemetry unavailable")

        print(f"Search area: width={self._area.width_m:.2f}m length={self._area.length_m:.2f}m row_spacing={self._area.row_spacing_m:.2f}m")
        print(f"Planned waypoints: {len(self._waypoints)}")

    async def calibrate_imu_biases(self, duration_s: float = 2.0) -> None:
        """Collect stationary IMU samples before arming."""
        print("Calibrating stationary IMU bias; keep the drone still")
        accel_samples: list[np.ndarray] = []
        gyro_samples: list[np.ndarray] = []
        started = time.monotonic()

        while time.monotonic() - started < duration_s:
            imu = self._latest_imu
            if imu is not None:
                accel_samples.append(imu.accel_body_m_s2.copy())
                gyro_samples.append(imu.gyro_body_rad_s.copy())
            await asyncio.sleep(0.02)

        self._estimator.set_stationary_bias(accel_samples, gyro_samples)
        print("IMU bias calibration complete")

    async def arm_drone(self) -> None:
        """Arm the drone through MAVSDK."""
        print("Arming drone")
        await self._drone.action.arm()

    async def takeoff_to_safe_height(self) -> None:
        """Take off to the configured scan altitude."""
        print(f"Setting takeoff altitude to {self._area.scan_altitude_m:.2f}m")
        await self._drone.action.set_takeoff_altitude(self._area.scan_altitude_m)
        print("Taking off")
        await self._drone.action.takeoff()
        await asyncio.sleep(8.0)

    async def start_offboard_with_hold(self) -> None:
        """Start offboard mode after pushing a zero-velocity setpoint."""
        print("Sending initial zero-velocity setpoint")
        await self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.2)
        print("Starting Offboard mode")
        try:
            await self._drone.offboard.start()
        except OffboardError as error:
            raise RuntimeError(f"Failed to start Offboard mode: {error}") from error
        self._offboard_started = True

    async def follow_waypoints_using_mavsdk(self) -> None:
        """Follow the generated snake path using body-frame velocity commands."""
        mission_started = time.monotonic()
        current_index = 1 if len(self._waypoints) > 1 else 0
        print("Starting snake traversal")

        while current_index < len(self._waypoints):
            if time.monotonic() - mission_started > self._args.max_mission_time_s:
                raise RuntimeError("Mission timeout exceeded")

            imu = self._latest_imu
            if imu is None:
                await asyncio.sleep(0.02)
                continue

            state = self._estimator.update(imu)
            if float(np.linalg.norm(state.position_m[:2])) > self._args.max_position_error_m:
                raise RuntimeError("Estimated position drift exceeded safety threshold")

            waypoint = self._waypoints[current_index]
            command = self._compute_body_velocity_command(state, waypoint)
            await self._drone.offboard.set_velocity_body(command)

            if self._waypoint_reached(state, waypoint):
                print(
                    f"Reached waypoint {current_index + 1}/{len(self._waypoints)} "
                    f"target=({waypoint.x_m:.2f}, {waypoint.y_m:.2f}, {waypoint.z_m:.2f}) "
                    f"est=({state.position_m[0]:.2f}, {state.position_m[1]:.2f}, {state.position_m[2]:.2f})"
                )
                await self.hold_position(self._args.hold_time_s)
                self._estimator.reset_horizontal_velocity()
                current_index += 1
                continue

            await asyncio.sleep(0.05)

        print("Snake traversal complete")
        await self.hold_position(self._args.hold_time_s)

    async def hold_position(self, hold_time_s: float) -> None:
        """Command a hover/hold by sending zero body velocity."""
        hold_command = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        deadline = time.monotonic() + hold_time_s
        while time.monotonic() < deadline:
            await self._drone.offboard.set_velocity_body(hold_command)
            await asyncio.sleep(0.05)

    async def stop_drone_safely(self) -> None:
        """Stop active motion without assuming full mission success."""
        print("Stopping active movement")
        try:
            await self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass

    async def land_or_disarm_safely(self) -> None:
        """Exit offboard if possible, then land and let PX4 manage descent."""
        await self.stop_drone_safely()

        if self._offboard_started:
            try:
                await self._drone.offboard.stop()
                print("Offboard mode stopped")
            except OffboardError as error:
                print(f"Offboard stop reported an error: {error}")
            except Exception as error:
                print(f"Unexpected error while stopping Offboard mode: {error}")

        try:
            print("Landing")
            await self._drone.action.land()
            await asyncio.sleep(8.0)
        except Exception as error:
            print(f"Landing command failed: {error}")
            try:
                print("Attempting disarm")
                await self._drone.action.disarm()
            except Exception as disarm_error:
                print(f"Disarm also failed: {disarm_error}")

    async def close_mavlink_connection(self) -> None:
        """Cancel background telemetry tasks."""
        for task in self._telemetry_tasks:
            task.cancel()
        if self._telemetry_tasks:
            await asyncio.gather(*self._telemetry_tasks, return_exceptions=True)
        self._telemetry_tasks.clear()

    async def _stream_imu(self) -> None:
        """Background task: keep the latest IMU sample cached."""
        try:
            async for imu in self._drone.telemetry.imu():
                accel = self._extract_accel_frd(imu)
                gyro = self._extract_gyro_frd(imu)
                self._latest_imu = ImuSample(
                    accel_body_m_s2=accel,
                    gyro_body_rad_s=gyro,
                    timestamp_s=time.monotonic(),
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"IMU telemetry stream stopped: {error}")

    async def _stream_battery(self) -> None:
        """Background task: cache battery state."""
        try:
            async for battery in self._drone.telemetry.battery():
                self._latest_battery_fraction = float(getattr(battery, "remaining_percent", 0.0))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"Battery telemetry stream stopped: {error}")

    def _compute_body_velocity_command(self, state: EstimatedState, waypoint: Waypoint) -> VelocityBodyYawspeed:
        """Compute a body-frame velocity command from waypoint error."""
        error_world = np.array(
            [
                waypoint.x_m - state.position_m[0],
                waypoint.y_m - state.position_m[1],
                0.0,
            ],
            dtype=np.float64,
        )
        distance_m = float(np.linalg.norm(error_world[:2]))
        if distance_m < 1e-6:
            return VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)

        desired_world_velocity = error_world / distance_m * min(distance_m, waypoint.max_speed_m_s)

        yaw = state.yaw_rad
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rot_body_from_world = np.array(
            [
                [cos_yaw, sin_yaw, 0.0],
                [-sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        desired_body_velocity = rot_body_from_world @ desired_world_velocity

        forward_m_s = float(np.clip(desired_body_velocity[0], -waypoint.max_speed_m_s, waypoint.max_speed_m_s))
        right_m_s = float(np.clip(desired_body_velocity[1], -waypoint.max_speed_m_s, waypoint.max_speed_m_s))
        down_m_s = 0.0
        yaw_rate_deg_s = 0.0
        return VelocityBodyYawspeed(forward_m_s, right_m_s, down_m_s, yaw_rate_deg_s)

    @staticmethod
    def _waypoint_reached(state: EstimatedState, waypoint: Waypoint) -> bool:
        """Return whether the estimated XY position is within the waypoint tolerance."""
        dx = waypoint.x_m - state.position_m[0]
        dy = waypoint.y_m - state.position_m[1]
        return math.hypot(dx, dy) <= waypoint.tolerance_m

    @staticmethod
    def _extract_accel_frd(imu_message: object) -> np.ndarray:
        """Best-effort extraction of FRD acceleration from MAVSDK telemetry."""
        accel = getattr(imu_message, "acceleration_frd", None)
        if accel is None:
            raise AttributeError("IMU message does not expose acceleration_frd")
        return np.array(
            [
                float(getattr(accel, "forward_m_s2")),
                float(getattr(accel, "right_m_s2")),
                float(getattr(accel, "down_m_s2")),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _extract_gyro_frd(imu_message: object) -> np.ndarray:
        """Best-effort extraction of FRD angular velocity from MAVSDK telemetry."""
        gyro = getattr(imu_message, "angular_velocity_frd", None)
        if gyro is None:
            raise AttributeError("IMU message does not expose angular_velocity_frd")
        return np.array(
            [
                float(getattr(gyro, "forward_rad_s")),
                float(getattr(gyro, "right_rad_s")),
                float(getattr(gyro, "down_rad_s")),
            ],
            dtype=np.float64,
        )


async def run_demo(args: argparse.Namespace) -> None:
    """Execute the full mission with safe shutdown handling around the mission body."""
    mission = MavsdkImuGridDemo(args)

    try:
        await mission.connect_to_pixhawk()
        await mission.run_preflight_checks()
        await mission.calibrate_imu_biases()
        await mission.arm_drone()
        await mission.takeoff_to_safe_height()
        await mission.start_offboard_with_hold()
        await mission.follow_waypoints_using_mavsdk()

    except KeyboardInterrupt:
        await mission.stop_drone_safely()
        print("Mission manually interrupted.")

    except Exception as error:
        await mission.stop_drone_safely()
        print(f"Mission failed: {error}")

    finally:
        await mission.land_or_disarm_safely()
        await mission.close_mavlink_connection()
        print("Mission terminated safely.")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
