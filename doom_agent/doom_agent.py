"""Jev DOOM Agent

A real-time DOOM controller powered by the local Jev decision engine.
Supports both:
  1. 'defend_the_center.cfg' - Continuous 360-degree survival arena with waves of demons
  2. 'basic.cfg' - Rapid single-target crosshair tracking
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import vizdoom as vzd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models import RunRequest, Step
from app.native_engine import NativeJevEngine


class JevDoomController:
    def __init__(
        self,
        scenario: str = "defend_the_center.cfg",
        window_visible: bool = True,
        save_gif: bool = True,
        max_episodes: int = 1,
    ):
        self.scenario = scenario
        self.window_visible = window_visible
        self.save_gif = save_gif
        self.max_episodes = max_episodes
        self.engine = NativeJevEngine()
        self.model_path = str(PROJECT_ROOT / "models" / "Qwen3-1.7B-Q8_0.gguf")

        # Resolve scenario config
        scenarios_dir = Path(vzd.__file__).parent / "scenarios"
        if not scenario.endswith(".cfg"):
            scenario = f"{scenario}.cfg"
        self.config_path = scenarios_dir / scenario
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Scenario not found: {self.config_path}")

        # Check if scenario is a turning (360) or strafing scenario
        self.is_turning = "defend" in scenario or "corridor" in scenario

    def create_game(self) -> vzd.DoomGame:
        game = vzd.DoomGame()
        game.load_config(str(self.config_path))
        game.set_window_visible(self.window_visible)
        game.set_objects_info_enabled(True)
        game.set_sectors_info_enabled(True)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.init()
        return game

    def build_prompt_state(self, game: vzd.DoomGame, state: vzd.GameState) -> Tuple[str, str]:
        player = None
        enemies = []

        for obj in (state.objects or []):
            if obj.name == "DoomPlayer":
                player = obj
            elif obj.name != "DoomPlayer" and not obj.name.startswith("Artifact"):
                enemies.append(obj)

        ammo = 50
        health = 100
        try:
            ammo = int(game.get_game_variable(vzd.GameVariable.AMMO2))
            health = int(game.get_game_variable(vzd.GameVariable.HEALTH))
        except Exception:
            pass

        if not player or not enemies:
            return "Room clear. Target neutralized.", "none"

        # Find closest threatening enemy
        closest_enemy = min(
            enemies,
            key=lambda e: math.hypot(e.position_x - player.position_x, e.position_y - player.position_y)
        )
        dx = closest_enemy.position_x - player.position_x
        dy = closest_enemy.position_y - player.position_y
        dist = math.hypot(dx, dy)

        if self.is_turning:
            target_angle = math.degrees(math.atan2(dy, dx))
            angle_diff = (target_angle - player.angle + 180) % 360 - 180
            if abs(angle_diff) <= 12.0:
                bearing_desc = f"DIRECTLY in the crosshairs (offset {angle_diff:+.1f} deg). Fire weapon now."
                alignment = "centered"
            elif angle_diff > 0:
                bearing_desc = f"{abs(angle_diff):.1f} degrees to your LEFT. Must turn left to face target."
                alignment = "left"
            else:
                bearing_desc = f"{abs(angle_diff):.1f} degrees to your RIGHT. Must turn right to face target."
                alignment = "right"
        else:
            # Linear strafing (basic.cfg)
            if abs(dy) <= 24.0:
                bearing_desc = "DIRECTLY centered in crosshairs. Line of fire clear. Fire weapon now."
                alignment = "centered"
            elif dy > 0:
                bearing_desc = f"{abs(dy):.1f} units to your LEFT. Must strafe left to center target."
                alignment = "left"
            else:
                bearing_desc = f"{abs(dy):.1f} units to your RIGHT. Must strafe right to center target."
                alignment = "right"

        state_text = (
            f"DOOM Tactical Sensor Data:\n"
            f"- Player Health: {health}%\n"
            f"- Ammo: {ammo}\n"
            f"- Closest Enemy: {closest_enemy.name} (distance {dist:.1f} units)\n"
            f"- Crosshair Alignment: {bearing_desc}\n"
        )
        return state_text, alignment

    def decide_action(self, state_text: str) -> Tuple[str, float]:
        left_action = "turn_left" if self.is_turning else "strafe_left"
        right_action = "turn_right" if self.is_turning else "strafe_right"

        request = RunRequest(
            context=state_text,
            mode="native",
            model_path=self.model_path,
            workflow=[
                Step(
                    id="action",
                    kind="choice",
                    prompt="Select the single tactical action to hit the target:",
                    options=["shoot_weapon", left_action, right_action],
                    criteria={
                        "shoot_weapon": "Shoot and fire weapon when target is directly in crosshairs",
                        left_action: f"{left_action.replace('_', ' ').capitalize()} to align crosshairs when target is on the left",
                        right_action: f"{right_action.replace('_', ' ').capitalize()} to align crosshairs when target is on the right"
                    }
                )
            ]
        )
        response = self.engine._run_sync(request)
        action = response.outputs.get("action", "shoot_weapon")
        confidence = 1.0
        if response.trace and response.trace[0].confidence is not None:
            confidence = response.trace[0].confidence
        return action, confidence

    def run(self):
        print("=" * 65)
        print("   Jev DOOM Slayer Agent (Local Qwen3 Native Logits Engine)")
        print(f"   Scenario: {self.scenario}")
        print(f"   Mode: {'360° Circular Arena' if self.is_turning else 'Linear Hallway'}")
        print(f"   Model: {Path(self.model_path).name}")
        print("=" * 65)

        game = self.create_game()
        left_action = "turn_left" if self.is_turning else "strafe_left"
        right_action = "turn_right" if self.is_turning else "strafe_right"

        actions_map = {
            left_action: [1, 0, 0],
            right_action: [0, 1, 0],
            "shoot_weapon": [0, 0, 1],
        }

        output_dir = Path(__file__).resolve().parent / "recordings"
        output_dir.mkdir(exist_ok=True)

        try:
            for episode in range(1, self.max_episodes + 1):
                print(f"\n>>> Starting Episode {episode} of {self.max_episodes}...")
                game.new_episode()
                frames = []
                step_idx = 0

                while not game.is_episode_finished():
                    state = game.get_state()
                    if state is None:
                        break

                    # Save screen frame for GIF if requested (cap at 150 frames to prevent huge files)
                    if self.save_gif and len(frames) < 150:
                        screen = state.screen_buffer
                        if screen.ndim == 3 and screen.shape[0] == 3:
                            img = Image.fromarray(screen.transpose(1, 2, 0))
                            frames.append(img)

                    # Extract state & query Jev
                    state_text, alignment = self.build_prompt_state(game, state)
                    t0 = time.perf_counter()
                    action_name, confidence = self.decide_action(state_text)
                    decision_ms = round((time.perf_counter() - t0) * 1000)

                    # Execute chosen action in DOOM (6 game tics = ~170ms game time)
                    action_vector = actions_map.get(action_name, [0, 0, 1])
                    reward = game.make_action(action_vector, 6)
                    step_idx += 1

                    symbol = "[FIRE!]" if action_name == "shoot_weapon" else ("[LEFT ]" if "left" in action_name else "[RIGHT]")
                    print(
                        f"  [Step {step_idx:03d} | {decision_ms:3d}ms] {symbol} {action_name:<13} "
                        f"(Conf: {confidence*100:5.1f}%) | Target: {alignment:<8} | Reward: {reward:+5.1f}"
                    )

                total_reward = game.get_total_reward()
                print(f"\n--- Episode {episode} Finished! Total Reward: {total_reward:+.1f} across {step_idx} steps ---")

                # Save episode GIF
                if self.save_gif and frames:
                    gif_path = output_dir / f"episode_{episode}.gif"
                    frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=120,
                        loop=0
                    )
                    print(f"  [Saved GIF] Recording saved to: recordings/{gif_path.name}")

                # Give user time to see final screen
                if self.window_visible:
                    time.sleep(2.0)

        finally:
            game.close()
            print("\nGame session ended. ViZDoom shut down cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jev DOOM Agent")
    parser.add_argument(
        "--scenario",
        default="defend_the_center.cfg",
        help="ViZDoom scenario: 'defend_the_center.cfg' (continuous arena) or 'basic.cfg' (quick shot)"
    )
    parser.add_argument("--no-window", action="store_true", help="Run in headless mode")
    parser.add_argument("--no-gif", action="store_true", help="Disable recording GIFs")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play")
    args = parser.parse_args()

    controller = JevDoomController(
        scenario=args.scenario,
        window_visible=not args.no_window,
        save_gif=not args.no_gif,
        max_episodes=args.episodes,
    )
    controller.run()
