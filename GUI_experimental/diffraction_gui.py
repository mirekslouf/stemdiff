"""
Diffraction Processing GUI Application
=======================================

A comprehensive GUI for processing electron diffraction data with multiple
reconstruction methods and theoretical profile comparison.

Author: David Rendulic
"""

import sys
import os
import traceback
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QGroupBox, QTabWidget, QFileDialog,
    QProgressBar, QTextEdit, QScrollArea, QSplitter, QMessageBox,
    QWizard, QWizardPage, QGridLayout, QFormLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class ProcessingThread(QThread):
    """Thread for running processing tasks without blocking the GUI"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.task_func(*self.args, **self.kwargs,
                                   progress_callback=self.emit_progress)
            self.finished.emit(result)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def emit_progress(self, value, message):
        self.progress.emit(value, message)


class MatplotlibWidget(QWidget):
    """Widget for displaying matplotlib figures"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self):
        self.figure.clear()
        self.canvas.draw()

    def plot_image(self, image, title="", cmap='viridis', vmax=None):
        """Plot a 2D image"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(image, cmap=cmap, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        self.figure.colorbar(im, ax=ax)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_line(self, x, y, xlabel="", ylabel="", title="", grid=True):
        """Plot a line graph"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if grid:
            ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_multiple_lines(self, data_list, xlabel="", ylabel="", title="", grid=True):
        """Plot multiple lines on the same graph

        Args:
            data_list: List of tuples (x, y, label, style)
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for x, y, label, style in data_list:
            ax.plot(x, y, style, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if grid:
            ax.grid(True, alpha=0.3)
        ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()


class Step1_DataLoadingPage(QWizardPage):
    """Step 1: Data Loading and Database Generation"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 1: Data Loading and Database Generation")
        self.setSubTitle("Select your data folder and configure database generation parameters")

        layout = QVBoxLayout()

        # Data folder selection
        folder_group = QGroupBox("Data Folder")
        folder_layout = QVBoxLayout()

        path_layout = QHBoxLayout()
        self.folder_path = QLineEdit()
        self.folder_path.setPlaceholderText("Select data folder containing .dat files...")
        self.folder_path.textChanged.connect(self.update_file_count)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        path_layout.addWidget(self.folder_path)
        path_layout.addWidget(browse_btn)
        folder_layout.addLayout(path_layout)

        # File pattern selection
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("File Pattern:"))
        self.file_pattern = QLineEdit()
        self.file_pattern.setText("*.dat")
        self.file_pattern.setToolTip("Pattern to match data files (e.g., *.dat, ??\*.dat, *_data.dat)")
        self.file_pattern.textChanged.connect(self.update_file_count)
        pattern_layout.addWidget(self.file_pattern)
        folder_layout.addLayout(pattern_layout)

        # File count display
        self.file_count_label = QLabel("Files found: 0")
        self.file_count_label.setStyleSheet("color: blue; font-weight: bold;")
        folder_layout.addWidget(self.file_count_label)

        # Preview files button
        preview_btn = QPushButton("Preview Files...")
        preview_btn.clicked.connect(self.preview_files)
        preview_btn.setMaximumWidth(150)
        folder_layout.addWidget(preview_btn)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # Parameters group
        params_group = QGroupBox("Database Parameters")
        params_layout = QFormLayout()

        self.sample_name = QLineEdit("Sample")
        self.num_files = QSpinBox()
        self.num_files.setRange(10, 10000)
        self.num_files.setValue(200)

        self.run_analysis = QCheckBox()
        self.run_analysis.setChecked(True)
        self.optimal_selection = QCheckBox()
        self.optimal_selection.setChecked(True)
        self.plot_examples = QCheckBox()
        self.plot_examples.setChecked(True)

        params_layout.addRow("Sample Name:", self.sample_name)
        params_layout.addRow("Number of Files:", self.num_files)
        params_layout.addRow("Run Analysis:", self.run_analysis)
        params_layout.addRow("Optimal File Selection:", self.optimal_selection)
        params_layout.addRow("Plot Examples:", self.plot_examples)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Progress and log
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        log_label = QLabel("Processing Log:")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)

        # Process buttons
        btn_layout = QHBoxLayout()

        self.preview_beam_btn = QPushButton("1. Preview Beam Positions")
        self.preview_beam_btn.clicked.connect(self.preview_beam_positions_before_db)
        self.preview_beam_btn.setStyleSheet("font-weight: bold; background-color: #e3f2fd;")
        btn_layout.addWidget(self.preview_beam_btn)

        process_btn = QPushButton("2. Generate Database with Current Thresholds")
        process_btn.clicked.connect(self.process_database)
        process_btn.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(process_btn)

        check_btn = QPushButton("Check Existing Database")
        check_btn.clicked.connect(self.check_existing_database)
        btn_layout.addWidget(check_btn)

        layout.addLayout(btn_layout)

        # Beam position thresholds (moved from Step 2 to Step 1)
        beam_group = QGroupBox("Beam Position Thresholds (for database generation)")
        beam_layout = QFormLayout()

        self.xcenter_min_db = QSpinBox()
        self.xcenter_min_db.setRange(0, 1024)
        self.xcenter_min_db.setValue(500)

        self.xcenter_max_db = QSpinBox()
        self.xcenter_max_db.setRange(0, 1024)
        self.xcenter_max_db.setValue(520)

        self.ycenter_min_db = QSpinBox()
        self.ycenter_min_db.setRange(0, 1024)
        self.ycenter_min_db.setValue(530)

        self.ycenter_max_db = QSpinBox()
        self.ycenter_max_db.setRange(0, 1024)
        self.ycenter_max_db.setValue(550)

        beam_layout.addRow("X Center Min:", self.xcenter_min_db)
        beam_layout.addRow("X Center Max:", self.xcenter_max_db)
        beam_layout.addRow("Y Center Min:", self.ycenter_min_db)
        beam_layout.addRow("Y Center Max:", self.ycenter_max_db)

        help_label = QLabel("💡 Click 'Preview Beam Positions' to see scatter plot and adjust these thresholds BEFORE generating database!")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: blue; font-style: italic; padding: 5px;")
        beam_layout.addRow(help_label)

        beam_group.setLayout(beam_layout)
        layout.addWidget(beam_group)

        # Results visualization section
        viz_group = QGroupBox("Results Visualization")
        viz_layout = QVBoxLayout()

        # Tabs for different visualizations
        self.viz_tabs = QTabWidget()

        # Tab 1: QC Plots
        self.qc_tab = QWidget()
        qc_layout = QVBoxLayout()
        self.qc_plot_widget = MatplotlibWidget()
        qc_layout.addWidget(self.qc_plot_widget)
        self.qc_tab.setLayout(qc_layout)
        self.viz_tabs.addTab(self.qc_tab, "QC Plots")

        # Tab 2: Sample Images
        self.samples_tab = QWidget()
        samples_layout = QVBoxLayout()

        # Intensity cut slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Intensity Cut:"))
        self.icut_slider = QSpinBox()
        self.icut_slider.setRange(0, 65535)
        self.icut_slider.setValue(300)
        self.icut_slider.valueChanged.connect(self.update_sample_images)
        slider_layout.addWidget(self.icut_slider)
        slider_layout.addStretch()
        samples_layout.addLayout(slider_layout)

        self.samples_plot_widget = MatplotlibWidget()
        samples_layout.addWidget(self.samples_plot_widget)
        self.samples_tab.setLayout(samples_layout)
        self.viz_tabs.addTab(self.samples_tab, "Sample Images")

        # Tab 3: PSF
        self.psf_tab = QWidget()
        psf_layout = QVBoxLayout()
        self.psf_plot_widget = MatplotlibWidget()
        psf_layout.addWidget(self.psf_plot_widget)
        self.psf_tab.setLayout(psf_layout)
        self.viz_tabs.addTab(self.psf_tab, "PSF")

        viz_layout.addWidget(self.viz_tabs)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        layout.addStretch()
        self.setLayout(layout)

        # Store results
        self.database_results = None
        self.sample_images = []  # Store sample images for updating

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.folder_path.setText(folder)
            self.update_file_count()

    def update_file_count(self):
        """Update the file count label based on current folder and pattern"""
        folder = self.folder_path.text()
        pattern = self.file_pattern.text()

        if not folder or not os.path.exists(folder):
            self.file_count_label.setText("Files found: 0 (invalid folder)")
            self.file_count_label.setStyleSheet("color: red; font-weight: bold;")
            return

        try:
            from pathlib import Path
            import glob

            # Use glob to find matching files
            full_pattern = os.path.join(folder, pattern)
            matching_files = glob.glob(full_pattern)
            count = len(matching_files)

            if count == 0:
                self.file_count_label.setText(f"Files found: 0 (pattern '{pattern}' matches no files)")
                self.file_count_label.setStyleSheet("color: red; font-weight: bold;")

                # Try to suggest alternative patterns
                all_dat_files = glob.glob(os.path.join(folder, "*.dat"))
                if all_dat_files:
                    self.file_count_label.setText(
                        f"Files found: 0 (pattern '{pattern}' matches nothing, "
                        f"but {len(all_dat_files)} .dat files exist. Try '*.dat')"
                    )
            else:
                self.file_count_label.setText(f"Files found: {count} ✓")
                self.file_count_label.setStyleSheet("color: green; font-weight: bold;")

        except Exception as e:
            self.file_count_label.setText(f"Files found: ? (error: {str(e)})")
            self.file_count_label.setStyleSheet("color: orange; font-weight: bold;")

    def preview_files(self):
        """Show a preview of files that match the pattern"""
        folder = self.folder_path.text()
        pattern = self.file_pattern.text()

        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "Error", "Please select a valid folder first")
            return

        try:
            import glob
            full_pattern = os.path.join(folder, pattern)
            matching_files = glob.glob(full_pattern)

            if not matching_files:
                QMessageBox.information(
                    self, "No Files Found",
                    f"No files match pattern '{pattern}' in:\n{folder}\n\n"
                    "Try a different pattern like '*.dat'"
                )
                return

            # Show first 20 files as preview
            preview_count = min(20, len(matching_files))
            file_names = [os.path.basename(f) for f in matching_files[:preview_count]]

            message = f"Found {len(matching_files)} files matching '{pattern}'\n\n"
            message += f"First {preview_count} files:\n"
            message += "\n".join(f"  • {name}" for name in file_names)

            if len(matching_files) > 20:
                message += f"\n\n... and {len(matching_files) - 20} more files"

            QMessageBox.information(self, "File Preview", message)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preview files:\n{str(e)}")

    def check_existing_database(self):
        """Check if database already exists and load it"""
        data_folder = self.folder_path.text()
        if not data_folder or not os.path.exists(data_folder):
            QMessageBox.warning(self, "Error", "Please select a valid data folder")
            return

        results_dir = os.path.join(data_folder, 'results')

        if not os.path.exists(results_dir):
            self.log("No results folder found. Need to generate database.")
            QMessageBox.information(self, "Not Found", "No existing database found. Please generate one.")
            return

        # Check for required files
        dbase_sum = os.path.join(results_dir, 'dbase_sum.zip')
        psf_file = os.path.join(results_dir, 'psf.npy')

        if not os.path.exists(dbase_sum):
            self.log("Database file not found. Need to generate database.")
            QMessageBox.information(self, "Not Found", "Database file not found. Please generate one.")
            return

        try:
            import stemdiff as sd
            import stemdiff.dbase

            self.log("Loading existing database...")

            # Load database
            df_sum = sd.dbase.read_database(dbase_sum)
            self.log(f"✓ Loaded database with {len(df_sum)} files")

            # Load PSF if exists
            psf = None
            if os.path.exists(psf_file):
                psf = np.load(psf_file)
                self.log("✓ Loaded PSF")

            # Setup SDATA and DIFFIMAGES
            SDATA = sd.gvars.SourceData(
                detector=sd.detectors.TimePix(),
                data_dir=data_folder,
                filenames=self.file_pattern.text()
            )
            DIFFIMAGES = sd.gvars.DiffImages()

            # Store results
            self.database_results = {
                'SDATA': SDATA,
                'DIFFIMAGES': DIFFIMAGES,
                'SDIR': results_dir,
                'df_sum': df_sum,
                'psf': psf,
                'data_folder': data_folder
            }

            # Load and display results
            self.load_visualization_results(results_dir, SDATA, df_sum, psf)

            QMessageBox.information(self, "Success",
                                  f"Loaded existing database with {len(df_sum)} files!")

        except Exception as e:
            error_msg = f"Error loading database: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to load database:\n{str(e)}")

    def preview_beam_positions_before_db(self):
        """Preview beam positions BEFORE applying filters - this helps set correct thresholds"""
        data_folder = self.folder_path.text()
        if not data_folder or not os.path.exists(data_folder):
            QMessageBox.warning(self, "Error", "Please select a valid data folder")
            return

        # Check if files exist with the pattern
        file_pattern = self.file_pattern.text()
        import glob
        matching_files = glob.glob(os.path.join(data_folder, file_pattern))

        if len(matching_files) == 0:
            QMessageBox.warning(self, "Error", f"No files found matching pattern '{file_pattern}'")
            return

        try:
            import stemdiff as sd
            import stemdiff.dbase
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel

            self.log("Calculating preliminary database (no filtering)...")
            self.progress_bar.setValue(10)

            # Setup SDATA
            SDATA = sd.gvars.SourceData(
                detector=sd.detectors.TimePix(),
                data_dir=data_folder,
                filenames=file_pattern
            )
            DIFFIMAGES = sd.gvars.DiffImages()

            self.progress_bar.setValue(30)

            # Calculate database WITHOUT filtering
            df_all = sd.dbase.calc_database(SDATA, DIFFIMAGES)

            if df_all is None or len(df_all) == 0:
                self.log("ERROR: No files could be read!")
                QMessageBox.critical(self, "Error", "No files could be read. Check file format.")
                self.progress_bar.setValue(0)
                return

            self.log(f"✓ Read {len(df_all)} files")
            self.progress_bar.setValue(100)

            # Show beam position scatter plot
            dialog = QDialog(self)
            dialog.setWindowTitle("Beam Position Preview (Before Filtering)")
            dialog.resize(900, 700)

            layout = QVBoxLayout()

            # Info label
            info = QLabel(
                f"<b>Total files: {len(df_all)}</b><br>"
                f"Adjust the thresholds in the main window to exclude outliers.<br>"
                f"Red box shows your current threshold settings."
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            plot_widget = MatplotlibWidget()

            # Extract data
            x_centers = df_all['Xcenter'].values
            y_centers = df_all['Ycenter'].values

            # Plot
            plot_widget.figure.clear()
            ax = plot_widget.figure.add_subplot(111)

            ax.scatter(x_centers, y_centers, alpha=0.6, s=20, c='blue',
                      label=f'{len(df_all)} files (unfiltered)')
            ax.set_xlabel('X Center', fontsize=12)
            ax.set_ylabel('Y Center', fontsize=12)
            ax.set_title('Beam Center Positions - ALL FILES (no filtering yet)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Draw current threshold box
            xcmin = self.xcenter_min_db.value()
            xcmax = self.xcenter_max_db.value()
            ycmin = self.ycenter_min_db.value()
            ycmax = self.ycenter_max_db.value()

            from matplotlib.patches import Rectangle
            rect = Rectangle((xcmin, ycmin), xcmax-xcmin, ycmax-ycmin,
                           linewidth=3, edgecolor='red', facecolor='none',
                           label='Current Threshold Box')
            ax.add_patch(rect)

            # Calculate how many would be kept
            inside = ((x_centers >= xcmin) & (x_centers <= xcmax) &
                     (y_centers >= ycmin) & (y_centers <= ycmax))
            n_inside = np.sum(inside)
            n_outside = len(df_all) - n_inside

            ax.legend(loc='best', fontsize=11)

            # Statistics box
            stats_text = (
                f'With current thresholds:\n'
                f'  KEPT: {n_inside} files ({100*n_inside/len(df_all):.1f}%)\n'
                f'  FILTERED OUT: {n_outside} files ({100*n_outside/len(df_all):.1f}%)\n\n'
                f'Current thresholds:\n'
                f'  X: {xcmin} - {xcmax}\n'
                f'  Y: {ycmin} - {ycmax}'
            )

            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Add recommendation
            if n_inside < 10:
                ax.text(0.5, 0.02, '⚠️ WARNING: Too few files will be kept! Loosen thresholds!',
                       transform=ax.transAxes, fontsize=11, color='red',
                       ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            elif n_inside < len(df_all) * 0.5:
                ax.text(0.5, 0.02, f'⚠️ Note: {100*n_outside/len(df_all):.0f}% of files will be filtered out',
                       transform=ax.transAxes, fontsize=10, color='orange',
                       ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            plot_widget.figure.tight_layout()
            plot_widget.canvas.draw()

            layout.addWidget(plot_widget)

            # Instructions
            instructions = QLabel(
                "<b>Next steps:</b><br>"
                "1. Adjust thresholds in main window to include the points you want<br>"
                "2. Click this button again to preview with new thresholds<br>"
                "3. When satisfied, click 'Generate Database with Current Thresholds'"
            )
            instructions.setWordWrap(True)
            layout.addWidget(instructions)

            dialog.setLayout(layout)
            dialog.exec_()

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to preview beam positions:\n{str(e)}")
            self.progress_bar.setValue(0)

    def load_visualization_results(self, results_dir, SDATA, df_sum, psf):
        """Load and display all visualization results"""
        try:
            # Load QC plots if they exist
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            sample_name = self.sample_name.text()

            # Try to load and display one of the QC plots
            qc_files = [
                f'Primary_beam_intensity_{sample_name}.png',
                f'XY_position_primary_beam_{sample_name}.png',
                f'Number_of_peaks_entropy_{sample_name}.png'
            ]

            for qc_file in qc_files:
                qc_path = os.path.join(results_dir, qc_file)
                if os.path.exists(qc_path):
                    img = mpimg.imread(qc_path)
                    self.qc_plot_widget.figure.clear()
                    ax = self.qc_plot_widget.figure.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')
                    self.qc_plot_widget.figure.tight_layout()
                    self.qc_plot_widget.canvas.draw()
                    break

            # Load and display sample images
            self.load_sample_images(SDATA, df_sum)

            # Display PSF if available
            if psf is not None:
                self.psf_plot_widget.plot_image(psf, title="Point Spread Function", cmap='hot', vmax=None)

        except Exception as e:
            self.log(f"Warning: Could not load some visualizations: {str(e)}")

    def load_sample_images(self, SDATA, df_sum):
        """Load 2 sample images from the database"""
        try:
            import stemdiff as sd
            from pathlib import Path

            # Get 2 sample files
            sample_count = min(2, len(df_sum))
            if sample_count == 0:
                return

            self.sample_images = []

            for idx in range(sample_count):
                datafile = df_sum.iloc[idx]
                # Handle both string and Path
                if isinstance(SDATA.data_dir, str):
                    datafile_name = Path(SDATA.data_dir) / datafile.DatafileName
                else:
                    datafile_name = SDATA.data_dir / datafile.DatafileName
                arr = sd.io.Datafiles.read(SDATA, datafile_name)
                self.sample_images.append(arr)

            self.log(f"Loaded {len(self.sample_images)} sample images")
            self.update_sample_images()

        except Exception as e:
            self.log(f"Warning: Could not load sample images: {str(e)}")

    def update_sample_images(self):
        """Update sample images display with current icut value"""
        if not self.sample_images:
            return

        try:
            icut = self.icut_slider.value()

            self.samples_plot_widget.figure.clear()

            num_samples = len(self.sample_images)
            for idx, img in enumerate(self.sample_images):
                ax = self.samples_plot_widget.figure.add_subplot(1, num_samples, idx + 1)

                # Apply icut
                img_display = np.clip(img, 0, icut)

                im = ax.imshow(img_display, cmap='viridis', vmax=icut)
                ax.set_title(f'Sample {idx + 1}\n(icut={icut})')
                ax.axis('off')
                self.samples_plot_widget.figure.colorbar(im, ax=ax, fraction=0.046)

            self.samples_plot_widget.figure.tight_layout()
            self.samples_plot_widget.canvas.draw()

        except Exception as e:
            self.log(f"Warning: Could not update sample images: {str(e)}")

    def log(self, message):
        self.log_text.append(message)

    def process_database(self):
        """Generate database from data files"""
        data_folder = self.folder_path.text()
        if not data_folder or not os.path.exists(data_folder):
            QMessageBox.warning(self, "Error", "Please select a valid data folder")
            return

        # Check if files exist with the pattern
        file_pattern = self.file_pattern.text()
        import glob
        matching_files = glob.glob(os.path.join(data_folder, file_pattern))

        if len(matching_files) == 0:
            QMessageBox.warning(
                self, "Error",
                f"No files found matching pattern '{file_pattern}' in selected folder.\n\n"
                f"Please adjust the file pattern. Common patterns:\n"
                f"  *.dat - all .dat files\n"
                f"  ??\*.dat - files starting with 2 characters\n"
                f"  data_*.dat - files starting with 'data_'"
            )
            return

        self.log(f"Found {len(matching_files)} files matching pattern '{file_pattern}'")
        self.log("Starting database generation...")
        self.progress_bar.setValue(10)

        try:
            # Import required modules
            import stemdiff as sd
            import stemdiff.dbase

            self.log("Loading data...")
            SDATA = sd.gvars.SourceData(
                detector=sd.detectors.TimePix(),
                data_dir=data_folder,
                filenames=file_pattern  # Use the configurable pattern
            )

            DIFFIMAGES = sd.gvars.DiffImages()
            SDIR = os.path.join(data_folder, 'results')
            os.makedirs(SDIR, exist_ok=True)

            self.progress_bar.setValue(30)
            self.log("Calculating database...")

            df1 = sd.dbase.calc_database(SDATA, DIFFIMAGES)

            if df1 is None or len(df1) == 0:
                self.log("ERROR: Database calculation returned 0 files!")
                self.log("This might mean:")
                self.log("  - File pattern doesn't match any files")
                self.log("  - Files are not readable")
                self.log("  - Files are not in TimePix format")
                self.progress_bar.setValue(0)
                QMessageBox.critical(
                    self, "Error",
                    "Database calculation failed - no files processed.\n\n"
                    "Please check:\n"
                    "1. File pattern matches your .dat files\n"
                    "2. Files are readable\n"
                    "3. Files are in correct TimePix format\n\n"
                    f"Searched for: {file_pattern} in {data_folder}"
                )
                return

            self.log(f"Database calculated: {len(df1)} files")

            self.progress_bar.setValue(50)

            # Save full database
            sd.dbase.save_database(df1, output_file=os.path.join(SDIR, 'dbase_all.zip'))
            self.log("Saved full database")

            # Analysis plots
            if self.run_analysis.isChecked():
                self.log("Generating analysis plots...")
                sd.io.set_plot_parameters(size=(14, 10), fontsize=11)

                # Primary beam intensity
                plot = df1.plot.line(y=['MaxInt'], color='green')
                plot.set_xlabel('Datafiles')
                plot.set_ylabel('Primary beam intensity')
                plot.grid()
                plt.savefig(os.path.join(SDIR, f'Primary_beam_intensity_{self.sample_name.text()}.png'))
                plt.close()

                # XY position
                plot = df1.plot.line(y=['Xcenter', 'Ycenter'])
                plot.set_xlabel('Datafiles')
                plot.set_ylabel('XY-position of primary beam')
                plot.grid()
                plt.savefig(os.path.join(SDIR, f'XY_position_primary_beam_{self.sample_name.text()}.png'))
                plt.close()

                # Peaks vs entropy
                plot = df1.plot.scatter(x='Peaks', y='S', color='red', marker='x')
                plot.set_xlabel('Number of peaks')
                plot.set_ylabel('Shannon entropy')
                plot.grid()
                plt.savefig(os.path.join(SDIR, f'Number_of_peaks_entropy_{self.sample_name.text()}.png'))
                plt.close()

                self.log("Analysis plots saved")

            self.progress_bar.setValue(70)

            # Optimal file selection
            if self.optimal_selection.isChecked():
                self.log("Selecting optimal files...")

                # Use GUI thresholds
                xcmin = self.xcenter_min_db.value()
                xcmax = self.xcenter_max_db.value()
                ycmin = self.ycenter_min_db.value()
                ycmax = self.ycenter_max_db.value()

                self.log(f"Applying beam position filter: X=[{xcmin}, {xcmax}], Y=[{ycmin}, {ycmax}]")

                df1b = df1[(df1.Peaks > 0)]
                df1b = df1b[(df1.MaxInt > 1000)]

                # Apply beam position filters with GUI values
                df1b = df1b[(xcmin < df1b.Xcenter) & (df1b.Xcenter < xcmax)]
                df1b = df1b[(ycmin < df1b.Ycenter) & (df1b.Ycenter < ycmax)]

                self.log(f"After beam position filtering: {len(df1)} → {len(df1b)} files")

                if len(df1b) == 0:
                    self.log("ERROR: All files filtered out by beam position thresholds!")
                    QMessageBox.critical(
                        self, "Error",
                        f"All files were filtered out!\n\n"
                        f"Current beam position thresholds:\n"
                        f"  X: {xcmin} - {xcmax}\n"
                        f"  Y: {ycmin} - {ycmax}\n\n"
                        f"Click 'Preview Beam Positions' to see actual positions\n"
                        f"and adjust thresholds."
                    )
                    self.progress_bar.setValue(0)
                    return

                df2 = df1b.sort_values(by=['Peaks', 'S'], ascending=[False, False])[:self.num_files.value()]
            else:
                df2 = df1.sort_values(by=['Peaks', 'S'], ascending=[False, False])[:self.num_files.value()]

            sd.dbase.save_database(df2, output_file=os.path.join(SDIR, 'dbase_sum.zip'))
            self.log(f"Selected {len(df2)} optimal files")

            self.progress_bar.setValue(85)

            # PSF calculation
            self.log("Calculating PSF...")
            df3 = df1[(df1.Peaks == 1) & (df1.MaxInt > 9000)]
            df3 = df3.sort_values(by='S', ascending=False)[-20:]

            import psf_function
            psf_guess = psf_function.PSFtype1.get_psf(SDATA, DIFFIMAGES, df3)
            psf_function.save_psf_to_disk(psf_guess, os.path.join(SDIR, 'psf.npy'))

            self.log("PSF calculated and saved")

            self.progress_bar.setValue(100)
            self.log("✓ Database generation complete!")

            # Store results
            self.database_results = {
                'SDATA': SDATA,
                'DIFFIMAGES': DIFFIMAGES,
                'SDIR': SDIR,
                'df_sum': df2,
                'psf': psf_guess,
                'data_folder': data_folder
            }

            # Load and display visualizations
            self.load_visualization_results(SDIR, SDATA, df2, psf_guess)

            QMessageBox.information(self, "Success", "Database generation completed successfully!")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Database generation failed:\n{str(e)}")
            self.progress_bar.setValue(0)

    def validatePage(self):
        """Validate before moving to next page"""
        if self.database_results is None:
            QMessageBox.warning(self, "Warning", "Please generate the database first")
            return False
        return True


class Step2_ProcessingPage(QWizardPage):
    """Step 2: Reconstruction Method Selection and Processing"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 2: Reconstruction and Processing")
        self.setSubTitle("Select reconstruction method and configure parameters")

        main_layout = QVBoxLayout()

        # Optional Additional Beam Position Filtering
        beam_group = QGroupBox("Optional: Additional Beam Position Filtering (for this run only)")
        beam_layout = QVBoxLayout()

        info_label = QLabel(
            "💡 Database was already filtered in Step 1.\n"
            "Use this ONLY if you want additional filtering for THIS processing run.\n"
            "Leave unchecked to use all files from the database."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-style: italic;")
        beam_layout.addWidget(info_label)

        # Checkbox to enable/disable filtering
        self.use_beam_filter = QCheckBox("Apply additional beam position filtering for this run")
        self.use_beam_filter.setChecked(False)  # Default to FALSE since database is already filtered
        beam_layout.addWidget(self.use_beam_filter)

        # Beam position thresholds
        thresh_layout = QFormLayout()

        self.xcenter_min = QSpinBox()
        self.xcenter_min.setRange(0, 1024)
        self.xcenter_min.setValue(500)

        self.xcenter_max = QSpinBox()
        self.xcenter_max.setRange(0, 1024)
        self.xcenter_max.setValue(520)

        self.ycenter_min = QSpinBox()
        self.ycenter_min.setRange(0, 1024)
        self.ycenter_min.setValue(530)

        self.ycenter_max = QSpinBox()
        self.ycenter_max.setRange(0, 1024)
        self.ycenter_max.setValue(550)

        thresh_layout.addRow("X Center Min:", self.xcenter_min)
        thresh_layout.addRow("X Center Max:", self.xcenter_max)
        thresh_layout.addRow("Y Center Min:", self.ycenter_min)
        thresh_layout.addRow("Y Center Max:", self.ycenter_max)

        beam_layout.addLayout(thresh_layout)

        # Button to visualize beam positions FROM DATABASE
        viz_beam_btn = QPushButton("Visualize Beam Positions (from loaded database)")
        viz_beam_btn.clicked.connect(self.visualize_beam_positions)
        beam_layout.addWidget(viz_beam_btn)

        beam_group.setLayout(beam_layout)
        main_layout.addWidget(beam_group)

        # Method selection
        method_group = QGroupBox("Reconstruction Method")
        method_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        methods = [
            "raw_sum - No processing",
            "RL_global_psf - Richardson-Lucy with global PSF",
            "GMM_fit - Gaussian Mixture Model segmentation",
            "Row_thresholding - Polar coordinate segmentation",
            "PeakFinding - Blob detection (LoG/DoH/MSER/PCBR)",
            "UNet - Neural network segmentation"
        ]
        self.method_combo.addItems(methods)
        self.method_combo.currentIndexChanged.connect(self.update_parameter_panel)

        method_layout.addWidget(QLabel("Select Method:"))
        method_layout.addWidget(self.method_combo)
        method_group.setLayout(method_layout)
        main_layout.addWidget(method_group)

        # Parameters panel (will be dynamically updated)
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_widget.setLayout(self.params_layout)
        self.params_scroll.setWidget(self.params_widget)
        main_layout.addWidget(self.params_scroll)

        # Output Configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout()

        self.output_bit_depth = QComboBox()
        self.output_bit_depth.addItems(["16-bit", "8-bit"])

        self.output_icut = QSpinBox()
        self.output_icut.setRange(0, 65535)
        self.output_icut.setValue(300)

        output_layout.addRow("Bit Depth:", self.output_bit_depth)
        output_layout.addRow("Intensity Cut (icut):", self.output_icut)

        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        # Progress
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        main_layout.addWidget(self.log_text)

        # Process button
        process_btn = QPushButton("Run Processing")
        process_btn.clicked.connect(self.run_processing)
        main_layout.addWidget(process_btn)

        self.setLayout(main_layout)

        # Initialize parameter panel
        self.update_parameter_panel()

        # Store results
        self.processing_results = None

    def update_parameter_panel(self):
        """Update parameter panel based on selected method"""
        # Clear existing parameters
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        method_idx = self.method_combo.currentIndex()

        if method_idx == 0:  # raw_sum
            label = QLabel("No additional parameters required")
            self.params_layout.addWidget(label)

        elif method_idx == 1:  # RL_global_psf
            self.setup_rl_parameters()

        elif method_idx == 2:  # GMM_fit
            label = QLabel("No additional parameters required (automatic)")
            self.params_layout.addWidget(label)

        elif method_idx == 3:  # Row_thresholding
            self.setup_row_threshold_parameters()

        elif method_idx == 4:  # PeakFinding
            self.setup_peak_finding_parameters()

        elif method_idx == 5:  # UNet
            self.setup_unet_parameters()

        self.params_layout.addStretch()

    def setup_rl_parameters(self):
        """Setup Richardson-Lucy parameters"""
        group = QGroupBox("Richardson-Lucy Parameters")
        layout = QFormLayout()

        self.rl_iterations = QSpinBox()
        self.rl_iterations.setRange(1, 100)
        self.rl_iterations.setValue(10)

        self.rl_regularization = QComboBox()
        self.rl_regularization.addItems(["None", "TM (Tikhonov-Miller)", "TV (Total Variation)"])

        self.rl_lambda = QDoubleSpinBox()
        self.rl_lambda.setRange(0.001, 1.0)
        self.rl_lambda.setValue(0.05)
        self.rl_lambda.setDecimals(3)
        self.rl_lambda.setSingleStep(0.01)

        layout.addRow("Iterations:", self.rl_iterations)
        layout.addRow("Regularization:", self.rl_regularization)
        layout.addRow("Lambda (reg. parameter):", self.rl_lambda)

        group.setLayout(layout)
        self.params_layout.addWidget(group)

    def setup_row_threshold_parameters(self):
        """Setup row thresholding parameters"""
        # Warning about hardcoded parameters
        warning_group = QGroupBox("⚠ Important Note")
        warning_layout = QVBoxLayout()
        warning_label = QLabel(
            "These parameters are HARDCODED in reconstruct.py (line 46)\n"
            "and cannot be changed from this GUI.\n\n"
            "Current values in reconstruct.py:\n"
            "• centroid_power=60, threshold_factor=5.5\n"
            "• start_row=50, center_radius=40\n\n"
            "To modify these, edit reconstruct.py directly."
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: orange; padding: 5px;")
        warning_layout.addWidget(warning_label)
        warning_group.setLayout(warning_layout)
        self.params_layout.addWidget(warning_group)

        # Show reference values (disabled)
        group = QGroupBox("Reference Values (hardcoded in reconstruct.py)")
        layout = QFormLayout()

        self.rt_start_row = QSpinBox()
        self.rt_start_row.setRange(0, 200)
        self.rt_start_row.setValue(50)  # Actual value from reconstruct.py
        self.rt_start_row.setEnabled(False)

        self.rt_threshold_factor = QDoubleSpinBox()
        self.rt_threshold_factor.setRange(0.1, 10.0)
        self.rt_threshold_factor.setValue(5.5)  # Actual value from reconstruct.py
        self.rt_threshold_factor.setDecimals(2)
        self.rt_threshold_factor.setEnabled(False)

        self.rt_center_radius = QSpinBox()
        self.rt_center_radius.setRange(10, 200)
        self.rt_center_radius.setValue(40)  # Actual value from reconstruct.py
        self.rt_center_radius.setEnabled(False)

        self.rt_min_blob_size = QSpinBox()
        self.rt_min_blob_size.setRange(0, 100)
        self.rt_min_blob_size.setValue(30)  # Default from utilities
        self.rt_min_blob_size.setEnabled(False)

        self.rt_centroid_power = QDoubleSpinBox()
        self.rt_centroid_power.setRange(1.0, 100.0)
        self.rt_centroid_power.setValue(60.0)  # Actual value from reconstruct.py
        self.rt_centroid_power.setDecimals(1)
        self.rt_centroid_power.setEnabled(False)

        layout.addRow("Start Row:", self.rt_start_row)
        layout.addRow("Threshold Factor:", self.rt_threshold_factor)
        layout.addRow("Center Radius:", self.rt_center_radius)
        layout.addRow("Min Blob Size:", self.rt_min_blob_size)
        layout.addRow("Centroid Power:", self.rt_centroid_power)

        group.setLayout(layout)
        self.params_layout.addWidget(group)

    def setup_peak_finding_parameters(self):
        """Setup peak finding parameters"""
        # Warning about hardcoded parameters
        warning_group = QGroupBox("⚠ Important Note")
        warning_layout = QVBoxLayout()
        warning_label = QLabel(
            "These parameters are HARDCODED in reconstruct.py (line 47)\n"
            "and cannot be changed from this GUI.\n\n"
            "Current values in reconstruct.py:\n"
            "• peak_min_sigma=10, peak_max_sigma=40\n"
            "• num_sigma_steps=20, bg_disk_radius=70\n\n"
            "Detector and background methods are also hardcoded.\n"
            "To modify these, edit reconstruct.py directly."
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: orange; padding: 5px;")
        warning_layout.addWidget(warning_label)
        warning_group.setLayout(warning_layout)
        self.params_layout.addWidget(warning_group)

        group = QGroupBox("Reference Values (hardcoded in reconstruct.py)")
        layout = QFormLayout()

        self.pf_detector = QComboBox()
        self.pf_detector.addItems(['log', 'doh', 'mser', 'pcbr', 'vote'])
        self.pf_detector.setCurrentText('doh')  # Not used, hardcoded in reconstruct.py
        self.pf_detector.setEnabled(False)

        self.pf_background = QComboBox()
        self.pf_background.addItems(['opening', 'opening_downsampled', 'gaussian',
                                     'median', 'rolling_ball', 'polynomial'])
        self.pf_background.setCurrentText('opening_downsampled')  # Not used
        self.pf_background.setEnabled(False)

        self.pf_threshold_factor = QDoubleSpinBox()
        self.pf_threshold_factor.setRange(0.1, 10.0)
        self.pf_threshold_factor.setValue(3.5)
        self.pf_threshold_factor.setDecimals(1)
        self.pf_threshold_factor.setEnabled(False)

        self.pf_center_ignore = QSpinBox()
        self.pf_center_ignore.setRange(0, 200)
        self.pf_center_ignore.setValue(50)
        self.pf_center_ignore.setEnabled(False)

        self.pf_min_sigma = QSpinBox()
        self.pf_min_sigma.setRange(1, 50)
        self.pf_min_sigma.setValue(10)  # Actual value from reconstruct.py
        self.pf_min_sigma.setEnabled(False)

        self.pf_max_sigma = QSpinBox()
        self.pf_max_sigma.setRange(10, 100)
        self.pf_max_sigma.setValue(40)  # Actual value from reconstruct.py
        self.pf_max_sigma.setEnabled(False)

        self.pf_bg_disk_radius = QSpinBox()
        self.pf_bg_disk_radius.setRange(10, 200)
        self.pf_bg_disk_radius.setValue(70)  # Actual value from reconstruct.py
        self.pf_bg_disk_radius.setEnabled(False)

        layout.addRow("Detector Method:", self.pf_detector)
        layout.addRow("Background Method:", self.pf_background)
        layout.addRow("Threshold Factor:", self.pf_threshold_factor)
        layout.addRow("Center Ignore Radius:", self.pf_center_ignore)
        layout.addRow("Min Sigma (LoG/DoH):", self.pf_min_sigma)
        layout.addRow("Max Sigma (LoG/DoH):", self.pf_max_sigma)
        layout.addRow("BG Disk Radius:", self.pf_bg_disk_radius)

        group.setLayout(layout)
        self.params_layout.addWidget(group)

    def setup_unet_parameters(self):
        """Setup UNet parameters"""
        # Warning about hardcoded parameters
        warning_group = QGroupBox("⚠ Important Note")
        warning_layout = QVBoxLayout()
        warning_label = QLabel(
            "Checkpoint path is HARDCODED in reconstruct.py (line 49)\n"
            "and cannot be changed from this GUI.\n\n"
            "Current path in reconstruct.py:\n"
            "C:\\Users\\drend\\OneDrive\\Plocha\\VU\\pythonProject\\\n"
            "NN_models\\DP_checkpoint_last.ckpt\n\n"
            "To use a different checkpoint, edit reconstruct.py directly."
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: orange; padding: 5px; font-family: monospace; font-size: 9pt;")
        warning_layout.addWidget(warning_label)
        warning_group.setLayout(warning_layout)
        self.params_layout.addWidget(warning_group)

        group = QGroupBox("Reference Values (hardcoded in reconstruct.py)")
        layout = QFormLayout()

        self.unet_checkpoint = QLineEdit()
        self.unet_checkpoint.setText(r"C:\Users\drend\OneDrive\Plocha\VU\pythonProject\NN_models\DP_checkpoint_last.ckpt")
        self.unet_checkpoint.setEnabled(False)  # Not used, hardcoded in reconstruct.py

        browse_btn = QPushButton("Browse...")
        browse_btn.setEnabled(False)  # Disabled since it won't work

        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(self.unet_checkpoint)
        checkpoint_layout.addWidget(browse_btn)

        layout.addRow("Checkpoint Path:", checkpoint_layout)

        group.setLayout(layout)
        self.params_layout.addWidget(group)

    def browse_file(self, line_edit, file_filter):
        """Browse for a file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if filename:
            line_edit.setText(filename)

    def log(self, message):
        self.log_text.append(message)

    def visualize_beam_positions(self):
        """Visualize beam positions from database to help set thresholds"""
        step1 = self.wizard().page(0)
        if not hasattr(step1, 'database_results') or step1.database_results is None:
            QMessageBox.warning(self, "Error", "Please complete Step 1 first")
            return

        try:
            df_sum = step1.database_results['df_sum']

            if len(df_sum) == 0:
                QMessageBox.warning(self, "Error", "Database is empty")
                return

            # Create a popup window with plot
            from PyQt5.QtWidgets import QDialog, QVBoxLayout

            dialog = QDialog(self)
            dialog.setWindowTitle("Beam Position Distribution")
            dialog.resize(800, 600)

            layout = QVBoxLayout()
            plot_widget = MatplotlibWidget()

            # Plot scatter of beam positions
            plot_widget.figure.clear()
            ax = plot_widget.figure.add_subplot(111)

            # Extract X and Y center data
            x_centers = df_sum['Xcenter'].values
            y_centers = df_sum['Ycenter'].values

            # Plot all points
            ax.scatter(x_centers, y_centers, alpha=0.6, s=20, c='blue', label=f'{len(df_sum)} frames')
            ax.set_xlabel('X Center', fontsize=12)
            ax.set_ylabel('Y Center', fontsize=12)
            ax.set_title(f'Beam Center Positions ({len(df_sum)} files)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Draw current threshold box
            xcmin = self.xcenter_min.value()
            xcmax = self.xcenter_max.value()
            ycmin = self.ycenter_min.value()
            ycmax = self.ycenter_max.value()

            from matplotlib.patches import Rectangle
            rect = Rectangle((xcmin, ycmin), xcmax-xcmin, ycmax-ycmin,
                           linewidth=2, edgecolor='red', facecolor='none',
                           label='Current Threshold Box')
            ax.add_patch(rect)

            # Show how many points are inside vs outside
            inside = ((x_centers >= xcmin) & (x_centers <= xcmax) &
                     (y_centers >= ycmin) & (y_centers <= ycmax))
            n_inside = np.sum(inside)
            n_outside = len(df_sum) - n_inside

            ax.legend(loc='best')
            ax.text(0.02, 0.98, f'Inside box: {n_inside}\nOutside box: {n_outside}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plot_widget.figure.tight_layout()
            plot_widget.canvas.draw()

            layout.addWidget(plot_widget)
            dialog.setLayout(layout)
            dialog.exec_()

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to visualize beam positions:\n{str(e)}")

    def run_processing(self):
        """Run the reconstruction processing"""
        # Get database results from previous page
        step1 = self.wizard().page(0)
        if not hasattr(step1, 'database_results') or step1.database_results is None:
            QMessageBox.warning(self, "Error", "Please complete Step 1 first")
            return

        db_results = step1.database_results

        self.log("Starting reconstruction processing...")
        self.progress_bar.setValue(10)

        try:
            # Import required modules
            import reconstruct  # The reconstruction module
            from utilities import (DiffractionPeakSegmenter, DiffractionPeakFinder,
                                  UnetSegmenter)

            # Apply beam position filtering if enabled
            df_sum = db_results['df_sum'].copy()

            if self.use_beam_filter.isChecked():
                self.log("Applying beam position filtering...")
                original_count = len(df_sum)

                # Fix: Use consistent DataFrame reference to avoid reindexing warning
                mask = (
                    (df_sum['Xcenter'] > self.xcenter_min.value()) &
                    (df_sum['Xcenter'] < self.xcenter_max.value()) &
                    (df_sum['Ycenter'] > self.ycenter_min.value()) &
                    (df_sum['Ycenter'] < self.ycenter_max.value())
                )
                df_sum = df_sum[mask]

                filtered_count = len(df_sum)
                self.log(f"Filtered: {original_count} → {filtered_count} files")

                if filtered_count == 0:
                    QMessageBox.warning(self, "Error",
                                      "Beam position filtering removed all files! "
                                      "Adjust thresholds or disable filtering.")
                    self.progress_bar.setValue(0)
                    return

            method_idx = self.method_combo.currentIndex()
            method_map = {
                0: 'raw_sum',
                1: 'RL_global_psf',
                2: 'GMM_fit',
                3: 'Row_thresholding',  # Note: reconstruct.py calls this "Segment_polar_threshold"
                4: 'PeakFinding',
                5: 'UNet'
            }

            method = method_map[method_idx]
            self.log(f"Selected method: {method}")

            # NOTE: reconstruct.py initializes processors with HARDCODED parameters at the top of sum_datafiles()
            # The GUI parameters below are NOT used by reconstruct.py in its current form
            # To customize these parameters, you must edit reconstruct.py directly

            # Prepare kwargs - only parameters that reconstruct.py actually accepts
            kwargs = {}

            if method == 'RL_global_psf':
                kwargs['iterate'] = self.rl_iterations.value()
                reg_text = self.rl_regularization.currentText()
                if "None" in reg_text:
                    kwargs['regularization'] = None
                elif "TM" in reg_text:
                    kwargs['regularization'] = 'TM'
                elif "TV" in reg_text:
                    kwargs['regularization'] = 'TV'
                kwargs['lambda_reg'] = self.rl_lambda.value()
                kwargs['psf'] = db_results['psf']

            # For other methods, reconstruct.py uses hardcoded parameters
            # See lines 44-49 in reconstruct.py:
            # - Row_thresholding: DiffractionPeakSegmenter(verbose=False, centroid_power=60, threshold_factor=5.5, start_row=50, center_radius=40)
            # - PeakFinding: DiffractionPeakFinder(peak_min_sigma=10, peak_max_sigma=40, num_sigma_steps=20, bg_disk_radius=70)
            # - UNet: UnetSegmenter(checkpoint_path=r"C:\Users\drend\...\DP_checkpoint_last.ckpt")

            self.log("")
            self.log("⚠ NOTE: Parameters for Row_thresholding, PeakFinding, and UNet")
            self.log("  are hardcoded in reconstruct.py and cannot be changed from GUI.")
            self.log("  To modify these, edit lines 44-49 in reconstruct.py")
            self.log("")

            self.progress_bar.setValue(30)

            # Run the reconstruction
            self.log("Running reconstruction (this may take several minutes)...")

            result = reconstruct.sum_datafiles(
                db_results['SDATA'],
                db_results['DIFFIMAGES'],
                df_sum,
                method=method,
                **kwargs
            )

            self.progress_bar.setValue(90)

            # Save result with configured options
            import stemdiff as sd

            bit_depth = self.output_bit_depth.currentText()
            icut = self.output_icut.value()

            # Apply icut
            result_clipped = np.clip(result, 0, icut)

            # Convert to 8-bit if requested
            if "8-bit" in bit_depth:
                result_clipped = (result_clipped / icut * 255).astype(np.uint8)
                itype = '8bit'
            else:
                itype = '16bit'

            output_path = os.path.join(db_results['SDIR'], f'{method}_result.png')
            sd.io.Arrays.save_as_image(result_clipped, output_image=output_path, itype=itype)

            self.log(f"Result saved to: {output_path}")
            self.log(f"Settings: {bit_depth}, icut={icut}")
            self.progress_bar.setValue(100)
            self.log("✓ Processing complete!")

            # Store results
            self.processing_results = {
                'result': result,
                'result_clipped': result_clipped,
                'method': method,
                'output_path': output_path,
                'icut': icut,
                'bit_depth': bit_depth,
                **db_results
            }

            QMessageBox.information(self, "Success",
                                  f"Processing completed!\nResult saved to:\n{output_path}")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Processing failed:\n{str(e)}")
            self.progress_bar.setValue(0)

    def validatePage(self):
        """Validate before moving to next page"""
        if self.processing_results is None:
            QMessageBox.warning(self, "Warning", "Please run processing first")
            return False
        return True


class Step3_ProfilePage(QWizardPage):
    """Step 3: Profile Calculation and Theoretical Comparison"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 3: Profile Calculation and Comparison")
        self.setSubTitle("Calculate radial profile and compare with theoretical PXRD")

        # Create main splitter for visualization and controls
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # CIF file selection for theoretical calculation
        cif_group = QGroupBox("Theoretical PXRD Calculation")
        cif_layout = QVBoxLayout()

        cif_file_layout = QHBoxLayout()
        self.cif_path = QLineEdit()
        self.cif_path.setPlaceholderText("Select CIF file for theoretical profile...")
        cif_browse_btn = QPushButton("Browse CIF...")
        cif_browse_btn.clicked.connect(self.browse_cif)
        cif_file_layout.addWidget(self.cif_path)
        cif_file_layout.addWidget(cif_browse_btn)
        cif_layout.addLayout(cif_file_layout)

        # PXRD parameters
        pxrd_params = QFormLayout()

        self.pxrd_wavelength = QDoubleSpinBox()
        self.pxrd_wavelength.setRange(0.1, 2.0)
        self.pxrd_wavelength.setValue(0.71)
        self.pxrd_wavelength.setDecimals(3)

        self.pxrd_temp_factors = QDoubleSpinBox()
        self.pxrd_temp_factors.setRange(0.1, 2.0)
        self.pxrd_temp_factors.setValue(0.8)
        self.pxrd_temp_factors.setDecimals(2)

        self.pxrd_peak_sigma = QDoubleSpinBox()
        self.pxrd_peak_sigma.setRange(0.001, 0.1)
        self.pxrd_peak_sigma.setValue(0.01)
        self.pxrd_peak_sigma.setDecimals(3)
        self.pxrd_peak_sigma.setSingleStep(0.001)

        pxrd_params.addRow("Wavelength (Å):", self.pxrd_wavelength)
        pxrd_params.addRow("Temperature Factors:", self.pxrd_temp_factors)
        pxrd_params.addRow("Peak Sigma:", self.pxrd_peak_sigma)

        cif_layout.addLayout(pxrd_params)
        cif_group.setLayout(cif_layout)
        left_layout.addWidget(cif_group)

        # Background correction parameters
        bg_group = QGroupBox("Background Correction Parameters")
        bg_layout = QFormLayout()

        self.bg_pixel_range_min = QSpinBox()
        self.bg_pixel_range_min.setRange(0, 1000)
        self.bg_pixel_range_min.setValue(0)

        self.bg_pixel_range_max = QSpinBox()
        self.bg_pixel_range_max.setRange(0, 1000)
        self.bg_pixel_range_max.setValue(300)

        self.bg_xlim_max = QSpinBox()
        self.bg_xlim_max.setRange(0, 1000)
        self.bg_xlim_max.setValue(300)

        self.bg_ylim_max = QSpinBox()
        self.bg_ylim_max.setRange(0, 1000)
        self.bg_ylim_max.setValue(140)

        bg_layout.addRow("Radial Pixel Min:", self.bg_pixel_range_min)
        bg_layout.addRow("Radial Pixel Max:", self.bg_pixel_range_max)
        bg_layout.addRow("Plot X Limit:", self.bg_xlim_max)
        bg_layout.addRow("Plot Y Limit:", self.bg_ylim_max)

        bg_group.setLayout(bg_layout)
        left_layout.addWidget(bg_group)

        # Calibration parameters
        calib_group = QGroupBox("Calibration")
        calib_layout = QFormLayout()

        self.fine_tuning = QDoubleSpinBox()
        self.fine_tuning.setRange(0.1, 2.0)
        self.fine_tuning.setValue(1.0)
        self.fine_tuning.setDecimals(3)
        self.fine_tuning.setSingleStep(0.001)

        calib_layout.addRow("Fine Tuning Factor:", self.fine_tuning)
        calib_group.setLayout(calib_layout)
        left_layout.addWidget(calib_group)

        # Processing buttons
        btn_layout = QVBoxLayout()

        calc_radial_btn = QPushButton("Calculate Radial Profile")
        calc_radial_btn.clicked.connect(self.calculate_radial_profile)
        btn_layout.addWidget(calc_radial_btn)

        calc_pxrd_btn = QPushButton("Calculate Theoretical PXRD")
        calc_pxrd_btn.clicked.connect(self.calculate_pxrd)
        btn_layout.addWidget(calc_pxrd_btn)

        bg_corr_btn = QPushButton("Interactive Background Correction")
        bg_corr_btn.clicked.connect(self.background_correction)
        btn_layout.addWidget(bg_corr_btn)

        compare_btn = QPushButton("Generate Comparison Plot")
        compare_btn.clicked.connect(self.generate_comparison)
        btn_layout.addWidget(compare_btn)

        left_layout.addLayout(btn_layout)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        left_layout.addWidget(self.log_text)

        left_layout.addStretch()
        left_widget.setLayout(left_layout)

        # Right panel: Visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        self.plot_widget = MatplotlibWidget()
        right_layout.addWidget(self.plot_widget)

        right_widget.setLayout(right_layout)

        # Add to splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        # Set main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        # Store intermediate results
        self.radial_profile = None
        self.pxrd_profile = None
        self.bg_corrected_data = None

    def browse_cif(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select CIF File", "", "CIF Files (*.cif)")
        if filename:
            self.cif_path.setText(filename)

    def log(self, message):
        self.log_text.append(message)

    def calculate_radial_profile(self):
        """Calculate radial profile from experimental data"""
        step2 = self.wizard().page(1)
        if not hasattr(step2, 'processing_results') or step2.processing_results is None:
            QMessageBox.warning(self, "Error", "Please complete Step 2 first")
            return

        try:
            import ediff.radial

            self.log("Calculating radial profile...")
            result_image = step2.processing_results['result']

            profile = ediff.radial.calc_radial_distribution(result_image)
            self.radial_profile = profile

            # Plot
            self.plot_widget.plot_line(
                profile[0], profile[1],
                xlabel='Distance from center [pixels]',
                ylabel='Intensity',
                title='Experimental ED Radial Profile'
            )

            # Save
            sdir = step2.processing_results['SDIR']
            output_file = os.path.join(sdir, 'ed_radial_profile.txt')
            np.savetxt(output_file, np.transpose(profile),
                      fmt=['%4d', '%8.2f'],
                      header='Columns: Pixels, Intensity')

            self.log(f"✓ Radial profile calculated and saved to: {output_file}")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to calculate radial profile:\n{str(e)}")

    def calculate_pxrd(self):
        """Calculate theoretical PXRD profile"""
        cif_file = self.cif_path.text()
        if not cif_file or not os.path.exists(cif_file):
            QMessageBox.warning(self, "Error", "Please select a valid CIF file")
            return

        try:
            import ediff.pxrd
            import ediff.io

            self.log("Calculating theoretical PXRD profile...")

            # Set plot parameters
            ediff.io.set_plot_parameters(size=(10, 5), dpi=120, fontsize=8)

            # Calculate PXRD
            XTAL = ediff.pxrd.Crystal(
                structure=cif_file,
                temp_factors=self.pxrd_temp_factors.value()
            )

            EPAR = ediff.pxrd.Experiment(
                wavelength=self.pxrd_wavelength.value(),
                two_theta_range=(2, 140)
            )

            PPAR = ediff.pxrd.PlotParameters(
                x_axis='q',
                xlim=(0.5, 8.5)
            )

            PXDR = ediff.pxrd.PXRDcalculation(
                XTAL, EPAR, PPAR,
                peak_profile_sigma=self.pxrd_peak_sigma.value()
            )

            # Save
            step2 = self.wizard().page(1)
            sdir = step2.processing_results['SDIR']
            xrd_file = os.path.join(sdir, 'pxrd_theoretical.txt')
            PXDR.save_diffractogram(xrd_file)

            # Load and plot
            xrd = np.loadtxt(xrd_file, unpack=True)
            self.pxrd_profile = xrd

            self.plot_widget.plot_line(
                xrd[2], xrd[3],
                xlabel='q [Å⁻¹]',
                ylabel='Intensity',
                title='Theoretical PXRD Profile'
            )

            self.log(f"✓ Theoretical PXRD calculated and saved to: {xrd_file}")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to calculate PXRD:\n{str(e)}")

    def background_correction(self):
        """Run interactive background correction"""
        if self.radial_profile is None:
            QMessageBox.warning(self, "Error", "Please calculate radial profile first")
            return

        try:
            import ediff.background
            import ediff.io

            self.log("Starting interactive background correction...")
            self.log("Note: A new window will open. Close it when finished.")

            # Save raw profile temporarily
            step2 = self.wizard().page(1)
            sdir = step2.processing_results['SDIR']
            temp_file = os.path.join(sdir, 'temp_raw_profile.txt')
            output_file = os.path.join(sdir, 'ed_profile_bgcorrected.txt')

            np.savetxt(temp_file, np.transpose(self.radial_profile),
                      fmt=['%4d', '%8.2f'],
                      header='Columns: Pixels, Intensity')

            # Run interactive background correction
            data = ediff.background.InputData(temp_file, usecols=[0, 1], unpack=True)
            ppar = ediff.background.PlotParams(
                output_file, 'Pixels', 'Intensity',
                xlim=[self.bg_pixel_range_min.value(), self.bg_xlim_max.value()],
                ylim=[0, self.bg_ylim_max.value()]
            )

            iplot = ediff.background.InteractivePlot(data, ppar, CLI=False, messages=True)
            iplot.run()

            # Load corrected data if saved
            if os.path.exists(output_file):
                bg_corrected = np.loadtxt(output_file, unpack=True)
                self.bg_corrected_data = bg_corrected
                self.log(f"✓ Background corrected data saved to: {output_file}")

                # Plot corrected profile
                self.plot_widget.plot_line(
                    bg_corrected[0], bg_corrected[2],
                    xlabel='Distance [pixels]',
                    ylabel='Intensity',
                    title='Background Corrected ED Profile'
                )
            else:
                self.log("Background correction cancelled or not saved")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Background correction failed:\n{str(e)}")

    def generate_comparison(self):
        """Generate final comparison plot"""
        if self.bg_corrected_data is None:
            QMessageBox.warning(self, "Error", "Please perform background correction first")
            return

        if self.pxrd_profile is None:
            QMessageBox.warning(self, "Error", "Please calculate theoretical PXRD first")
            return

        try:
            self.log("Generating comparison plot...")

            # Get corrected experimental data
            eld_pixels = self.bg_corrected_data[0]
            eld_intensity = self.bg_corrected_data[2]

            # Get theoretical data
            q_theory = self.pxrd_profile[2]
            i_theory = self.pxrd_profile[3]

            # Calibrate experimental data
            # Find max intensity in theory
            max_idx_theory = np.argmax(i_theory)
            q_at_max = q_theory[max_idx_theory]

            # Find max in experimental
            max_idx_exp = np.argmax(eld_intensity)
            pixel_at_max = eld_pixels[max_idx_exp]

            # Calibration constant
            if pixel_at_max > 0:
                calib_constant = q_at_max / pixel_at_max
            else:
                calib_constant = 0.01

            self.log(f"Calibration constant: {calib_constant:.4f} q/pixel")

            # Apply calibration
            q_exp = eld_pixels * calib_constant * self.fine_tuning.value()

            # Normalize experimental
            max_exp = np.max(eld_intensity)
            if max_exp > 0:
                i_exp_norm = eld_intensity / max_exp
            else:
                i_exp_norm = eld_intensity

            # Create comparison plot
            data_list = [
                (q_theory, i_theory, f'PXRD (Theoretical)', 'b-'),
                (q_exp, i_exp_norm, 'ED (Experimental, normalized)', 'r-')
            ]

            self.plot_widget.plot_multiple_lines(
                data_list,
                xlabel='q [Å⁻¹]',
                ylabel='Intensity',
                title='Comparison: Theoretical PXRD vs Experimental ED',
                grid=True
            )

            # Save comparison data
            step2 = self.wizard().page(1)
            sdir = step2.processing_results['SDIR']

            # Interpolate to common q grid
            q_common = np.linspace(max(q_theory.min(), q_exp.min()),
                                  min(q_theory.max(), q_exp.max()),
                                  500)

            i_theory_interp = np.interp(q_common, q_theory, i_theory, left=0, right=0)
            i_exp_interp = np.interp(q_common, q_exp, i_exp_norm, left=0, right=0)

            comparison_file = os.path.join(sdir, 'comparison_ed_pxrd.txt')
            np.savetxt(comparison_file,
                      np.transpose([q_common, i_theory_interp, i_exp_interp]),
                      header='q[A^-1]\tIntensity_Theory\tIntensity_Exp_Normalized',
                      fmt='%.6f', delimiter='\t')

            self.log(f"✓ Comparison data saved to: {comparison_file}")

            # Also save the plot
            plot_file = os.path.join(sdir, 'comparison_plot.png')
            self.plot_widget.figure.savefig(plot_file, dpi=120, facecolor='white')
            self.log(f"✓ Comparison plot saved to: {plot_file}")

            QMessageBox.information(self, "Success",
                                  f"Comparison generated successfully!\n"
                                  f"Files saved to:\n{sdir}")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            QMessageBox.critical(self, "Error", f"Comparison generation failed:\n{str(e)}")


class DiffractionProcessingWizard(QWizard):
    """Main wizard for diffraction processing"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Diffraction Data Processing - Complete Pipeline")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.HaveHelpButton, False)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)

        # Set minimum size
        self.resize(1200, 800)

        # Add pages
        self.addPage(Step1_DataLoadingPage())
        self.addPage(Step2_ProcessingPage())
        self.addPage(Step3_ProfilePage())

        # Style
        self.setStyleSheet("""
            QWizard {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show wizard
    wizard = DiffractionProcessingWizard()
    wizard.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()