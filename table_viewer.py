from PyQt6.QtWidgets import (QApplication, QMainWindow, QTableWidget, 
                           QTableWidgetItem, QVBoxLayout, QWidget, 
                           QTabWidget, QPushButton, QLabel, QScrollArea)
import pandas as pd
import sys
import os

class VesselDataViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Vessel Data Analysis Viewer')
        self.setGeometry(100, 100, 1600, 1000)  # Increased window size
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Define output directory
        output_dir = "output"
        
        # Dataset files to load with paths
        datasets = {
            'Original Dataset': os.path.join(output_dir, 'original_vessel_data.csv'),
            'Clean Dataset': os.path.join(output_dir, 'clean_vessel_data.csv'),
            'Suspicious Dataset': os.path.join(output_dir, 'suspicious_vessels.csv'),
            'Combined Dataset': os.path.join(output_dir, 'combined_vessel_data.csv')
        }
        
        # Create tabs for each dataset
        try:
            if not os.path.exists(output_dir):
                raise FileNotFoundError(f"Output directory '{output_dir}' not found")
            
            for title, file in datasets.items():
                if not os.path.exists(file):
                    raise FileNotFoundError(f"File not found: {file}")
                df = pd.read_csv(file)
                tab = self.create_table_tab(df, title)
                tabs.addTab(tab, title)
                
                # Add count label
                count_label = QLabel(f"Total Vessels in {title}: {len(df)}")
                layout.addWidget(count_label)
                
        except FileNotFoundError as e:
            error_label = QLabel(f"Error: {str(e)}\nPlease run cleaning.py first to generate the output files.")
            layout.addWidget(error_label)
    
    def create_distribution_text(self, df):
        """Create formatted text for distributions"""
        # Vessel Group Distribution
        vessel_dist = df['VesselGroup'].value_counts()
        vessel_pct = df['VesselGroup'].value_counts(normalize=True) * 100
        
        # Cargo Type Distribution
        cargo_dist = df['Cargo_type'].value_counts()
        cargo_pct = df['Cargo_type'].value_counts(normalize=True) * 100
        
        # Flag Distribution (top 10)
        flag_dist = df['Flag'].value_counts().head(10)
        flag_pct = df['Flag'].value_counts(normalize=True).head(10) * 100
        
        # Cargo Weight Statistics
        cargo_stats = df['Total_weight_cargo'].describe()
        
        # Create formatted text
        text = f"""
DATASET SUMMARY
==============

Total Vessels: {len(df)}

Vessel Group Distribution
------------------------
"""
        for vessel, count in vessel_dist.items():
            text += f"{vessel}: {count} ({vessel_pct[vessel]:.1f}%)\n"
        
        text += f"""
Cargo Type Distribution
----------------------
"""
        for cargo, count in cargo_dist.items():
            text += f"{cargo}: {count} ({cargo_pct[cargo]:.1f}%)\n"
        
        text += f"""
Top 10 Flag States
-----------------
"""
        for flag, count in flag_dist.items():
            text += f"{flag}: {count} ({flag_pct[flag]:.1f}%)\n"
        
        text += f"""
Cargo Weight Statistics
----------------------
Mean: {cargo_stats['mean']:.2f}
Median: {cargo_stats['50%']:.2f}
Min: {cargo_stats['min']:.2f}
Max: {cargo_stats['max']:.2f}
Std Dev: {cargo_stats['std']:.2f}

Destination Port Distribution (Top 10)
------------------------------------
"""
        dest_dist = df['Destination_port'].value_counts().head(10)
        dest_pct = df['Destination_port'].value_counts(normalize=True).head(10) * 100
        for port, count in dest_dist.items():
            text += f"{port}: {count} ({dest_pct[port]:.1f}%)\n"
        
        return text
    
    def create_table_tab(self, df, title):
        """Create a table widget with statistics for the given DataFrame"""
        # Create widget and scroll area
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create scroll area for statistics
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Add distribution statistics
        stats_text = self.create_distribution_text(df)
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-family: monospace;")  # Use monospace font for better formatting
        scroll_layout.addWidget(stats_label)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Create and set up table
        table = QTableWidget()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        
        # Set headers
        table.setHorizontalHeaderLabels(df.columns)
        
        # Populate table
        for i in range(len(df)):
            for j in range(len(df.columns)):
                value = str(df.iloc[i, j])
                item = QTableWidgetItem(value)
                table.setItem(i, j, item)
        
        # Add table
        layout.addWidget(table)
        
        # Set layout proportions
        layout.setStretch(0, 1)  # Statistics section
        layout.setStretch(1, 2)  # Table section
        
        # Resize columns to content
        table.resizeColumnsToContents()
        
        return tab

def main():
    app = QApplication(sys.argv)
    viewer = VesselDataViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 