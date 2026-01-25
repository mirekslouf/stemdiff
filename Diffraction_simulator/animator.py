"""
Diffraction Animation Module
Creates 3D animations of electron diffraction process
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

from generator import DiffractionGenerator


class DiffractionAnimator:
    """Create animations of the electron diffraction process"""

    def __init__(self):
        self.generator = DiffractionGenerator()

    def create_animation(
        self,
        material: str = 'Al',
        zone_axis: list = None,
        voltage: float = 200,
        duration: float = 10,
        fps: int = 30,
        num_electrons: int = 50,
        save_path: str = None,
        crystal_size: float = 1.0,
        fig_size: tuple = (10, 8),
        dpi: int = 100
    ):
        """
        Create a 3D animation showing electrons traveling through crystal

        Args:
            material: Material name
            zone_axis: Zone axis direction (default [0,0,1])
            voltage: Accelerating voltage in kV
            duration: Animation duration in seconds
            fps: Frames per second
            num_electrons: Number of electron traces
            save_path: Path to save animation (mp4 or gif)
            crystal_size: Size of crystal for visualization
            fig_size: Figure size in inches
            dpi: DPI for animation

        Returns:
            matplotlib animation object
        """
        if zone_axis is None:
            zone_axis = [0, 0, 1]

        print(f"\n🎬 Creating animation for {material} [{''.join(map(str, zone_axis))}]")
        print(f"   Duration: {duration}s @ {fps} fps")
        print(f"   Electrons: {num_electrons}")

        # Generate the actual diffraction pattern
        print("   Generating diffraction pattern...")
        pattern, _ = self.generator.generate(
            material=material,
            zone_axis=zone_axis,
            voltage=voltage,
            size=256,
            generate_mask=False
        )

        # Setup 3D scene
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Crystal dimensions
        crystal_length = crystal_size
        crystal_width = crystal_size * 0.8
        crystal_height = crystal_size * 0.8

        # Detector settings
        detector_distance = crystal_length * 2
        detector_size = crystal_width * 3

        # Initialize electron positions
        electron_positions = self._initialize_electrons(
            num_electrons, crystal_width, crystal_height, crystal_length
        )

        # Electron paths
        electron_paths = [[] for _ in range(num_electrons)]

        # Assign destinations (diffraction spots)
        destinations = self._calculate_destinations(
            num_electrons, detector_size, detector_distance
        )

        # Animation state
        frames = int(duration * fps)
        pattern_fade_start = int(frames * 0.7)  # Start fading pattern at 70%

        # Create pattern as texture for detector
        pattern_extent = [-detector_size/2, detector_size/2, -detector_size/2, detector_size/2]

        def update(frame):
            ax.clear()
            progress = frame / frames

            # Update electron positions
            for i in range(num_electrons):
                new_pos = self._update_electron_position(
                    electron_positions[i],
                    destinations[i],
                    crystal_length,
                    detector_distance,
                    progress
                )
                electron_positions[i] = new_pos
                electron_paths[i].append(new_pos)

                # Limit path history
                tail_length = min(30, int(frame * 1.5) + 5)
                if len(electron_paths[i]) > tail_length:
                    electron_paths[i] = electron_paths[i][-tail_length:]

            # Camera angle
            if progress < 0.7:
                elev = 10 + progress * 70
                azim = -90
            else:
                elev = 80
                azim = -90 + (progress - 0.7) * 20

            ax.view_init(elev=elev, azim=azim)

            # Draw crystal
            self._draw_crystal(ax, crystal_width, crystal_height, crystal_length)

            # Draw electron paths
            for path in electron_paths:
                if len(path) > 1:
                    path_array = np.array(path)
                    ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                           '-', color='cyan', alpha=0.8, lw=1.0)

            # Draw detector frame
            if progress > 0.4:
                self._draw_detector(ax, detector_size, detector_distance, progress)

            # Fade in the actual diffraction pattern on detector
            if frame >= pattern_fade_start:
                fade_progress = (frame - pattern_fade_start) / (frames - pattern_fade_start)
                fade_alpha = min(1.0, fade_progress * 2)  # Fade in over second half

                # Create meshgrid for pattern
                X, Y = np.meshgrid(
                    np.linspace(-detector_size/2, detector_size/2, pattern.shape[1]),
                    np.linspace(-detector_size/2, detector_size/2, pattern.shape[0])
                )
                Z = np.ones_like(X) * detector_distance

                # Apply pattern as colors with fade (ensure values in [0, 1])
                pattern_normalized = np.clip(pattern, 0, 1)
                pattern_alpha = np.clip(pattern_normalized * fade_alpha, 0, 1)

                # Create RGBA array
                colors = np.zeros((*pattern.shape, 4))
                colors[:, :, 0] = pattern_normalized  # Red channel
                colors[:, :, 1] = pattern_normalized  # Green channel
                colors[:, :, 2] = pattern_normalized  # Blue channel
                colors[:, :, 3] = pattern_alpha  # Alpha channel with fade

                # Ensure all values in valid range
                colors = np.clip(colors, 0, 1)

                # Draw pattern on detector surface
                ax.plot_surface(X, Y, Z, facecolors=colors,
                              rstride=1, cstride=1, shade=False, antialiased=True)
            elif progress > 0.5:
                # Show forming spots before full pattern fade-in
                spot_progress = (progress - 0.5) / 0.2
                self._draw_diffraction_spots(ax, detector_distance, spot_progress)

            # Set limits
            ax.set_xlim(-detector_size / 2, detector_size / 2)
            ax.set_ylim(-detector_size / 2, detector_size / 2)
            ax.set_zlim(-crystal_length * 1.5, detector_distance * 1.2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{material} [{"".join(map(str, zone_axis))}] @ {voltage} kV',
                        fontsize=12, fontweight='bold')

            # Add progress indicator
            if progress < 1.0:
                status = f"Progress: {int(progress * 100)}%"
                ax.text2D(0.05, 0.95, status, transform=ax.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            return []

        # Create animation
        print("   Generating frames...")
        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

        # Save or display
        if save_path:
            print(f"   Saving to {save_path}...")
            if save_path.endswith('.mp4'):
                try:
                    anim.save(save_path, writer='ffmpeg', dpi=dpi)
                except:
                    save_path = save_path.replace('.mp4', '.gif')
                    anim.save(save_path, writer='pillow', dpi=dpi)
            else:
                anim.save(save_path, writer='pillow', dpi=dpi)

            print(f"   ✅ Animation saved: {save_path}")
        else:
            plt.show()

        plt.close()
        return anim

    def _initialize_electrons(self, num, width, height, length):
        """Initialize electron positions in a grid"""
        positions = []
        grid_size = int(np.ceil(np.sqrt(num)))
        source_distance = length * 1.5

        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) < num:
                    offset_x = (np.random.random() - 0.5) * 0.1 * width
                    offset_y = (np.random.random() - 0.5) * 0.1 * height

                    x = (i / (grid_size - 1 or 1) - 0.5) * width * 0.8 + offset_x
                    y = (j / (grid_size - 1 or 1) - 0.5) * height * 0.8 + offset_y
                    z = -source_distance

                    positions.append([x, y, z])

        return positions

    def _calculate_destinations(self, num, detector_size, detector_distance):
        """Calculate where each electron will end up on the detector"""
        destinations = []

        for _ in range(num):
            # Most go to direct beam (center)
            if np.random.random() < 0.4:
                x, y = 0, 0
            else:
                # Others scattered around
                angle = np.random.random() * 2 * np.pi
                radius = np.random.random() * detector_size * 0.3
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

            destinations.append([x, y, detector_distance])

        return destinations

    def _update_electron_position(self, pos, dest, crystal_length,
                                  detector_distance, progress):
        """Update single electron position"""
        x, y, z = pos
        dest_x, dest_y, dest_z = dest

        # Before crystal: straight path
        if z < -crystal_length / 2:
            new_z = z + crystal_length * 0.15
            return [x, y, new_z]

        # Inside crystal: gradual shift
        elif z < crystal_length / 2:
            crystal_progress = (z - (-crystal_length / 2)) / crystal_length

            angle_x = np.arctan2(dest_x, detector_distance) * crystal_progress * 2
            angle_y = np.arctan2(dest_y, detector_distance) * crystal_progress * 2
            shift_x = angle_x * 0.1
            shift_y = angle_y * 0.1

            scatter_x = (np.random.random() - 0.5) * 0.02 * (1 - crystal_progress)
            scatter_y = (np.random.random() - 0.5) * 0.02 * (1 - crystal_progress)

            new_z = z + crystal_length * 0.1
            new_x = x + shift_x + scatter_x
            new_y = y + shift_y + scatter_y

            return [new_x, new_y, new_z]

        # After crystal: linear to destination
        else:
            remaining_z = detector_distance - z
            if remaining_z <= 0:
                return [dest_x, dest_y, dest_z]

            new_z = z + remaining_z * 0.1
            new_x = x + (dest_x - x) * 0.1
            new_y = y + (dest_y - y) * 0.1

            return [new_x, new_y, new_z]

    def _draw_crystal(self, ax, width, height, length):
        """Draw crystal as translucent box"""
        # Define vertices
        vertices = [
            [-width/2, -height/2, -length/2],
            [width/2, -height/2, -length/2],
            [width/2, height/2, -length/2],
            [-width/2, height/2, -length/2],
            [-width/2, -height/2, length/2],
            [width/2, -height/2, length/2],
            [width/2, height/2, length/2],
            [-width/2, height/2, length/2]
        ]

        # Define faces
        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [3, 2, 6, 7]]
        ]

        crystal = Poly3DCollection(faces, alpha=0.2, facecolor='lightblue',
                                  edgecolor='darkblue', linewidth=0.5)
        ax.add_collection3d(crystal)

    def _draw_detector(self, ax, size, distance, progress):
        """Draw detector plane"""
        fade = min(1.0, (progress - 0.4) / 0.2)

        # Frame
        frame_x = [-size/2, size/2, size/2, -size/2, -size/2]
        frame_y = [-size/2, -size/2, size/2, size/2, -size/2]
        frame_z = [distance] * 5

        ax.plot(frame_x, frame_y, frame_z, 'k-', lw=1, alpha=fade)

    def _draw_diffraction_spots(self, ax, distance, progress):
        """Draw diffraction spots on detector"""
        fade = min(1.0, (progress - 0.5) / 0.3)

        # Central spot
        ax.scatter(0, 0, distance, s=500*fade, c='yellow', alpha=fade,
                  edgecolor='white', linewidth=0.5)



if __name__ == '__main__':
    # Demo animation
    animator = DiffractionAnimator()

    animator.create_animation(
        material='Al',
        zone_axis=[1, 1, 1],
        voltage=200,
        duration=10,
        save_path='diffraction_animation.gif'
    )