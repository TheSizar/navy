import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
import sys
import numpy as np
from typing import Tuple, List
import os

# Second, all data structures
region_to_countries = {
    'Asia': ['China', 'Japan', 'South Korea', 'Singapore', 'Malaysia', 'India'],
    'North America': ['USA', 'Canada'],
    'Europe': ['Netherlands', 'Germany', 'Belgium', 'Spain', 'Italy'],
    'Middle East': ['UAE', 'Saudi Arabia', 'Qatar', 'Oman'],
    'Australia': ['Australia'],
    'Africa': ['South Africa', 'Nigeria', 'Egypt', 'Morocco'],
    'Americas': ['Brazil', 'Argentina', 'Mexico', 'Chile'],
    'West Africa': ['Nigeria', 'Angola', 'Ghana'],
    'Global': ['China', 'USA', 'Singapore', 'Netherlands', 'Japan']
}

# Major ports by country and region with annual cargo volume (million tons)
MAJOR_PORTS = {
    'China': {
        'East': {
            'Shanghai': 750,
            'Ningbo-Zhoushan': 700,
            'Qingdao': 550,
            'Tianjin': 500,
            'Dalian': 450
        },
        'South': {
            'Shenzhen': 650,
            'Guangzhou': 600,
            'Xiamen': 450,
            'Hong Kong': 400
        }
    },
    'USA': {
        'Gulf': {
            'Houston': 275,
            'South Louisiana': 235,
            'Corpus Christi': 150,
            'New Orleans': 160,
            'Beaumont': 100,
            'Mobile': 120
        },
        'East Coast': {
            'New York': 250,
            'Savannah': 150,
            'Charleston': 130,
            'Miami': 110,
            'Jacksonville': 100
        },
        'West Coast': {
            'Los Angeles': 300,
            'Long Beach': 280,
            'Oakland': 170,
            'Seattle': 140,
            'Tacoma': 130
        }
    },
    'Singapore': {
        'Main': {
            'Singapore': 600,
            'Jurong': 300
        }
    },
    'Netherlands': {
        'Main': {
            'Rotterdam': 450,
            'Amsterdam': 200
        }
    },
    'Japan': {
        'Main': {
            'Nagoya': 300,
            'Yokohama': 280,
            'Tokyo': 270,
            'Osaka': 250,
            'Kobe': 230
        }
    },
    'South Korea': {
        'Main': {
            'Busan': 400,
            'Ulsan': 300,
            'Incheon': 250
        }
    },
    'UAE': {
        'Main': {
            'Dubai': 350,
            'Abu Dhabi': 300,
            'Jebel Ali': 280
        }
    },
    'Saudi Arabia': {
        'Main': {
            'Jeddah': 300,
            'Dammam': 250,
            'Yanbu': 200
        }
    },
    'Germany': {
        'Main': {
            'Hamburg': 300,
            'Bremerhaven': 200,
            'Wilhelmshaven': 150
        }
    },
    'Belgium': {
        'Main': {
            'Antwerp': 280,
            'Zeebrugge': 150
        }
    },
    'Spain': {
        'Main': {
            'Valencia': 200,
            'Algeciras': 180,
            'Barcelona': 160
        }
    },
    'Malaysia': {
        'Main': {
            'Port Klang': 250,
            'Tanjung Pelepas': 200
        }
    },
    'Brazil': {
        'Main': {
            'Santos': 200,
            'Paranagua': 150,
            'Rio de Janeiro': 130
        }
    },
    'Australia': {
        'Main': {
            'Port Hedland': 500,  # Major iron ore port
            'Newcastle': 170,
            'Brisbane': 150
        }
    },
    'Canada': {
        'Main': {
            'Vancouver': 200,
            'Montreal': 150,
            'Prince Rupert': 120
        }
    },
    'Italy': {
        'Main': {
            'Genoa': 150,
            'Trieste': 130,
            'Gioia Tauro': 120
        }
    },
    'India': {
        'Main': {
            'Mumbai': 200,
            'Chennai': 180,
            'Mundra': 150
        }
    },
    'Russia': {
        'Main': {
            'Novorossiysk': 180,
            'St. Petersburg': 150,
            'Vladivostok': 120
        }
    }
}

PORT_SPECIALIZATIONS = {
    'CONTAINER': {
        'Shanghai': {'capacity': 47.3, 'type': 'TEU'},
        # ... other container ports
    },
    # ... other specializations
}

VESSEL_CARGO_PORT_MAPPING = {
    'Container Ship': {
        'port_type': 'CONTAINER',
        # ... other container ship mappings
    },
    # ... other vessel types
}

TRADE_ROUTES = {
    'major_container_routes': {
        'Asia-North America': {
            'volume': 28.3,
            # ... other route details
        },
        # ... other routes
    },
    # ... other route types
}

# Third, all functions
def get_port_country(port: str) -> str:
    """Get the country for a given port"""
    for country, regions in MAJOR_PORTS.items():
        for region, ports in regions.items():
            if port in ports:
                return country
    return "Unknown"

def assign_default_route() -> Tuple[str, str, str, str, str, float]:
    """Assign default route and cargo for vessels that don't match known types"""
    all_ports = []
    for country, regions in MAJOR_PORTS.items():
        for region, ports in regions.items():
            for port in ports.keys():
                all_ports.append((country, port))
    
    # Ensure we have valid ports to choose from
    if not all_ports:
        return ('Singapore', 'Singapore', 'China', 'Shanghai', 'General Cargo', 0.0)
    
    origin_idx = np.random.randint(0, len(all_ports))
    dest_idx = np.random.randint(0, len(all_ports))
    while dest_idx == origin_idx:  # Ensure different ports
        dest_idx = np.random.randint(0, len(all_ports))
    
    origin = all_ports[origin_idx]
    dest = all_ports[dest_idx]
    
    return (
        origin[0],
        origin[1],
        dest[0],
        dest[1],
        'General Cargo',
        np.random.uniform(1000, 5000)  # Reasonable default weight
    )

def load_vessel_type_mapping(mapping_file="inputs/VesselTypeCodes_Structured.csv"):
    """
    Load vessel type mapping from CSV file
    Returns two dictionaries: one for vessel groups and one for detailed classifications
    """
    try:
        mapping_df = pd.read_csv(mapping_file)
        
        # Create empty dictionaries for the mappings
        group_mapping = {}
        type_mapping = {}
        
        # Iterate through the mapping DataFrame to handle ranges
        for _, row in mapping_df.iterrows():
            code_str = str(row['AIS Vessel Code'])
            group = row['Vessel Group (2018)']
            type_detail = row['AIS Ship & Cargo Classification']
            
            # Handle ranges (e.g., "1-19")
            if '-' in code_str:
                start, end = map(int, code_str.split('-'))
                for code in range(start, end + 1):
                    group_mapping[code] = group
                    type_mapping[code] = type_detail
            else:
                # Handle single values
                try:
                    code = int(code_str)
                    group_mapping[code] = group
                    type_mapping[code] = type_detail
                except ValueError:
                    print(f"Warning: Could not convert '{code_str}' to integer")
        
        print("\nVessel type mapping loaded successfully")
        print(f"Total vessel codes mapped: {len(group_mapping)}")
        print("\nSample mappings:")
        sample_codes = list(group_mapping.keys())[:3]
        for code in sample_codes:
            print(f"Code {code}: Group={group_mapping[code]}, Type={type_mapping[code]}")
        
        return group_mapping, type_mapping
        
    except Exception as e:
        print(f"Error loading vessel type mapping: {str(e)}")
        raise

def load_and_clean_ais_data(file_path="inputs/AIS_2023_12_31.csv"):
    """Load and clean AIS data with vessel type mapping"""
    try:
        print("Loading data...")
        df = pd.read_csv(file_path)
        n_vessels_initial = len(df)
        
        print(f"Initial number of records: {n_vessels_initial}")
        
        # Remove rows where Vessel Name is NULL
        print("Removing records with NULL vessel names...")
        df = df.dropna(subset=['VesselName'])
        
        # Sort by MMSI and BaseDateTime to ensure we get the latest reading
        print("Sorting data by MMSI and timestamp...")
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        df = df.sort_values(['MMSI', 'BaseDateTime'])
        
        # Get the last reading for each vessel (MMSI)
        print("Filtering to latest position for each vessel...")
        df = df.drop_duplicates(subset=['MMSI'], keep='last')
        
        # Drop the 'Cargo' column if it exists
        if 'Cargo' in df.columns:
            df = df.drop(columns=['Cargo'])
        
        # Rest of the cleaning process...
        n_vessels = len(df)
        print(f"Number of unique vessels: {n_vessels}")
        
        # Load vessel type mappings
        vessel_group_mapping, vessel_type_mapping = load_vessel_type_mapping()
        
        print("Basic cleaning...")
        # Vectorized basic cleaning
        df['VesselType'] = df['VesselType'].astype(float).astype('Int64')
        df['VesselGroup'] = df['VesselType'].map(vessel_group_mapping)
        df['VesselTypeDetailed'] = df['VesselType'].map(vessel_type_mapping)
        
        # Convert VesselGroup to numpy array of strings
        vessel_groups = df['VesselGroup'].astype(str).values
        valid_mask = vessel_groups != 'Not Available'
        
        # Pre-allocate arrays for results
        origin_countries = np.full(n_vessels, 'Unknown', dtype=object)
        dest_countries = np.full(n_vessels, 'Unknown', dtype=object)
        origin_ports = np.full(n_vessels, 'Unknown', dtype=object)
        dest_ports = np.full(n_vessels, 'Unknown', dtype=object)
        cargo_types = np.full(n_vessels, 'Unknown', dtype=object)
        weights = np.zeros(n_vessels)
        
        # Convert lookup dictionaries to arrays for faster access
        all_countries = np.array(list(region_to_countries['Global']))
        
        print("Processing vessels in bulk...")
        # Process all valid vessels at once
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        if n_valid > 0:
            # Process each vessel individually for accurate cargo and weight assignment
            for idx in valid_indices:
                vessel_group = vessel_groups[idx]
                vessel_type_detailed = df['VesselTypeDetailed'].iloc[idx]
                
                # Use the assign_route_and_cargo function for each vessel
                (
                    origin_countries[idx],
                    origin_ports[idx],
                    dest_countries[idx],
                    dest_ports[idx],
                    cargo_types[idx],
                    weights[idx]
                ) = assign_route_and_cargo(vessel_group, vessel_type_detailed)
        
        print("Assigning results to DataFrame...")
        # Assign results back to DataFrame all at once
        df['Origin_country'] = origin_countries
        df['Destination_country'] = dest_countries
        df['Origin_port'] = origin_ports
        df['Destination_port'] = dest_ports
        df['Cargo_type'] = cargo_types
        df['Total_weight_cargo'] = weights
        df['Flag'] = df['VesselGroup'].apply(assign_flag)
        
        print("\nProcessing complete. Generating statistics...")
        print(f"Total vessels processed: {len(df)}")
        print(f"Valid vessels: {n_valid}")
        print("\nOrigin Country Distribution (top 5):")
        print(df['Origin_country'].value_counts().head())
        print("\nCargo Type Distribution (top 5):")
        print(df['Cargo_type'].value_counts().head())
        
        return df
        
    except Exception as e:
        print(f"Error in load_and_clean_ais_data: {str(e)}")
        raise

class DataFrameViewer(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle('AIS Data Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create the table widget
        self.table = QTableWidget()
        self.setCentralWidget(self.table)
        
        # Set the number of rows and columns
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        
        # Set the headers
        self.table.setHorizontalHeaderLabels(df.columns)
        
        # Populate the table
        for i in range(len(df)):
            for j in range(len(df.columns)):
                value = str(df.iloc[i, j])
                self.table.setItem(i, j, QTableWidgetItem(value))
        
        # Resize columns to content
        self.table.resizeColumnsToContents()

def calculate_cargo_weight(vessel_group: str, cargo_type: str, origin_port: str, dest_port: str) -> float:
    """Calculate realistic cargo weight based on vessel type and cargo"""
    base_weights = {
        'Container Ship': {
            'small': 30000,    # Feeder: 2,000-3,000 TEU
            'medium': 80000,   # Panamax: 4,000-8,000 TEU
            'large': 190000,   # ULCV: 18,000-24,000 TEU
            'variance': 0.15   # Less variance due to standardized containers
        },
        'Bulk Carrier': {
            'small': 45000,    # Handymax: 35,000-50,000 DWT
            'medium': 95000,   # Panamax: 65,000-100,000 DWT
            'large': 210000,   # Capesize: 150,000-400,000 DWT
            'variance': 0.10   # Bulk cargo is fairly consistent
        },
        'Oil Tanker': {
            'small': 85000,    # Aframax: 80,000-120,000 DWT
            'medium': 160000,  # Suezmax: 120,000-200,000 DWT
            'large': 320000,   # VLCC: 200,000-320,000 DWT
            'variance': 0.08   # Oil cargoes are very standardized
        },
        'LNG Tanker': {
            'small': 90000,    # Small scale: 70,000-120,000 m³
            'medium': 155000,  # Q-Flex: 150,000-170,000 m³
            'large': 266000,   # Q-Max: 260,000-270,000 m³
            'variance': 0.05   # Very standardized cargo
        },
        'Chemical Tanker': {
            'small': 25000,    # Small chemical tanker
            'medium': 40000,   # Medium chemical tanker
            'large': 55000,    # Large chemical tanker
            'variance': 0.12
        },
        'Vehicle Carrier': {
            'small': 4500,     # 4,500 vehicles
            'medium': 6500,    # 6,500 vehicles
            'large': 8500,     # 8,500 vehicles
            'variance': 0.10
        },
        'General Cargo Ship': {
            'small': 15000,    # Small general cargo
            'medium': 25000,   # Medium general cargo
            'large': 35000,    # Large general cargo
            'variance': 0.20
        },
        'Fishing': {
            'small': 50,       # Small fishing vessel
            'medium': 150,     # Medium fishing vessel
            'large': 300,      # Large fishing vessel
            'variance': 0.30
        },
        'Pleasure Craft': {
            'small': 0.5,      # Small pleasure craft
            'medium': 1.0,     # Medium pleasure craft
            'large': 2.0,      # Large pleasure craft
            'variance': 0.30
        }
    }
    
    # If vessel group not in base_weights, use General Cargo Ship as default
    if vessel_group not in base_weights:
        vessel_group = 'General Cargo Ship'
    
    # Select size based on port capacity
    port_size = min(
        MAJOR_PORTS.get(get_port_country(origin_port), {}).get('Main', {}).get(origin_port, 0),
        MAJOR_PORTS.get(get_port_country(dest_port), {}).get('Main', {}).get(dest_port, 0)
    )
    
    # Determine vessel size based on port capacity
    if port_size > 500:  # Major ports can handle largest vessels
        size = 'large'
    elif port_size > 250:  # Medium ports
        size = 'medium'
    else:  # Smaller ports
        size = 'small'
    
    weight_info = base_weights[vessel_group]
    base = weight_info[size]
    variance = weight_info['variance']
    
    # Adjust weight based on cargo type
    if cargo_type == 'Iron Ore':
        base *= 1.2  # Iron ore is denser
    elif cargo_type == 'Crude Oil':
        base *= 1.1  # Full oil cargoes
    elif cargo_type == 'Vehicles':
        base *= 0.9  # Vehicle carriers rarely sail full
    elif cargo_type in ['Electronics', 'Consumer Goods']:
        base *= 0.85  # Lighter cargo types
    elif cargo_type == 'Liquefied Natural Gas':
        base *= 0.95  # LNG specific adjustment
    
    # Generate weight with normal distribution
    weight = np.random.normal(base, base * variance)
    
    # Ensure weight is within realistic bounds
    min_weight = base * 0.3  # Minimum 30% of base weight
    max_weight = base * 1.3  # Maximum 130% of base weight
    weight = max(min_weight, min(max_weight, weight))
    
    return round(weight, 2)

def select_cargo_type(vessel_group: str, origin_country: str, dest_country: str) -> str:
    """Select cargo type based on vessel group and real-world cargo flows"""
    if vessel_group == 'Oil Tanker':
        if origin_country in ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Oman']:
            return np.random.choice(['Crude Oil', 'Petroleum Products'], p=[0.95, 0.05])
        elif origin_country in ['Singapore', 'South Korea', 'Netherlands']:
            return np.random.choice(['Petroleum Products', 'Fuel Oil'], p=[0.80, 0.20])
        return np.random.choice(['Crude Oil', 'Petroleum Products', 'Fuel Oil'], p=[0.70, 0.20, 0.10])
    
    elif vessel_group == 'Bulk Carrier':
        if origin_country == 'Australia':
            return np.random.choice(['Iron Ore', 'Coal', 'Bauxite'], p=[0.70, 0.25, 0.05])
        elif origin_country == 'Brazil':
            return np.random.choice(['Iron Ore', 'Soybeans', 'Grain'], p=[0.80, 0.15, 0.05])
        elif origin_country in ['USA', 'Canada']:
            return np.random.choice(['Grain', 'Soybeans', 'Coal'], p=[0.50, 0.30, 0.20])
        elif origin_country == 'South Africa':
            return np.random.choice(['Coal', 'Iron Ore', 'Manganese'], p=[0.60, 0.25, 0.15])
        return np.random.choice(['Coal', 'Grain', 'Iron Ore', 'Bauxite'], p=[0.35, 0.30, 0.25, 0.10])
    
    elif vessel_group == 'Container Ship':
        if origin_country in ['China', 'South Korea', 'Japan']:
            return np.random.choice([
                'Electronics', 'Consumer Goods', 'Auto Parts', 
                'Machinery', 'Textiles'
            ], p=[0.35, 0.25, 0.20, 0.15, 0.05])
        elif origin_country in ['Germany', 'Netherlands', 'Belgium']:
            return np.random.choice([
                'Machinery', 'Auto Parts', 'Chemical Products', 
                'Consumer Goods', 'Food Products'
            ], p=[0.35, 0.25, 0.20, 0.15, 0.05])
        return np.random.choice([
            'Consumer Goods', 'Machinery', 'Food Products',
            'Chemical Products', 'Mixed Cargo'
        ], p=[0.30, 0.25, 0.20, 0.15, 0.10])
    
    elif vessel_group == 'LNG Tanker':
        return 'Liquefied Natural Gas'
    
    elif vessel_group == 'Chemical Tanker':
        return np.random.choice([
            'Organic Chemicals', 'Inorganic Chemicals',
            'Vegetable Oils', 'Acids'
        ], p=[0.40, 0.30, 0.20, 0.10])
    
    elif vessel_group == 'Vehicle Carrier':
        return 'Vehicles'
    
    elif vessel_group == 'General Cargo Ship':
        # Enhanced cargo categories for general cargo ships based on real-world data
        if origin_country in ['China', 'South Korea', 'Japan', 'Taiwan']:
            return np.random.choice([
                'Steel Products', 'Industrial Equipment', 'Construction Materials',
                'Metal Products', 'Paper Products', 'Wood Products',
                'Agricultural Equipment', 'Project Cargo', 'Packaged Foods',
                'Recycling Materials'
            ], p=[0.20, 0.15, 0.15, 0.10, 0.10, 0.08, 0.08, 0.05, 0.05, 0.04])
        
        elif origin_country in ['Germany', 'Netherlands', 'Belgium', 'France']:
            return np.random.choice([
                'Industrial Machinery', 'Manufacturing Equipment', 'Steel Products',
                'Chemical Products', 'Construction Materials', 'Paper Products',
                'Agricultural Products', 'Project Cargo', 'Recycling Materials',
                'Specialized Equipment'
            ], p=[0.18, 0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.06, 0.05, 0.04])
        
        elif origin_country in ['USA', 'Canada']:
            return np.random.choice([
                'Agricultural Equipment', 'Construction Materials', 'Industrial Machinery',
                'Wood Products', 'Paper Products', 'Steel Products',
                'Chemical Products', 'Project Cargo', 'Recycling Materials',
                'Manufacturing Equipment'
            ], p=[0.15, 0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.05, 0.05])
        
        else:
            # Default distribution for other countries
            return np.random.choice([
                'Steel Products', 'Construction Materials', 'Industrial Equipment',
                'Agricultural Products', 'Wood Products', 'Paper Products',
                'Chemical Products', 'Project Cargo', 'Manufacturing Equipment',
                'Recycling Materials'
            ], p=[0.15, 0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.05, 0.05])
    
    # Default case (replacing 'Mixed Cargo' with specific cargo types)
    return np.random.choice([
        'Steel Products', 'Construction Materials', 'Industrial Equipment',
        'Agricultural Products', 'Wood Products', 'Paper Products',
        'Chemical Products', 'Project Cargo', 'Manufacturing Equipment',
        'Recycling Materials', 'Metal Products', 'Specialized Equipment',
        'Industrial Machinery', 'Packaged Foods', 'Textile Products'
    ], p=[0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.06, 0.06,
          0.05, 0.05, 0.03, 0.03, 0.02])

def map_vessel_group(vessel_group: str) -> str:
    """Map AIS vessel groups to standardized vessel types"""
    mapping = {
        'Cargo': 'Container Ship',  # Will be further refined based on vessel_type_detailed
        'Tanker': 'Oil Tanker',    # Will be further refined based on vessel_type_detailed
        'Fishing': 'Fishing',
        'Passenger': 'Passenger Ship',
        'Pleasure Craft/Sailing': 'Pleasure Craft',
        'Tug Tow': 'Tug',
        'Military': 'Military',
        'Other': 'Other'
    }
    return mapping.get(vessel_group, 'Other')

def refine_vessel_type(vessel_group: str, vessel_type_detailed: str) -> str:
    """Refine vessel type based on detailed classification"""
    if vessel_group == 'Cargo':
        if 'container' in vessel_type_detailed.lower():
            return 'Container Ship'
        elif 'bulk' in vessel_type_detailed.lower():
            return 'Bulk Carrier'
        elif 'vehicle' in vessel_type_detailed.lower():
            return 'Vehicle Carrier'
        return 'General Cargo Ship'
    
    elif vessel_group == 'Tanker':
        if 'chemical' in vessel_type_detailed.lower():
            return 'Chemical Tanker'
        elif 'lng' in vessel_type_detailed.lower() or 'gas' in vessel_type_detailed.lower():
            return 'LNG Tanker'
        return 'Oil Tanker'
    
    return map_vessel_group(vessel_group)

def assign_route_and_cargo(vessel_group: str, vessel_type_detailed: str) -> Tuple[str, str, str, str, str, float]:
    """Assign route and cargo based on vessel type and real-world trade patterns"""
    # First, refine the vessel type
    refined_vessel_type = refine_vessel_type(vessel_group, vessel_type_detailed)
    
    # Trade route probabilities based on vessel type
    trade_routes = {
        'Container Ship': {
            'Asia-North America': 0.45,
            'Asia-Europe': 0.30,
            'Europe-North America': 0.10,
            'Intra-Asia': 0.15
        },
        'Oil Tanker': {
            'Middle East-Asia': 0.40,
            'Middle East-Europe': 0.25,
            'Americas-Asia': 0.20,
            'Intra-Asia': 0.15
        },
        'Bulk Carrier': {
            'Australia-Asia': 0.35,
            'Brazil-Asia': 0.25,
            'North America-Asia': 0.20,
            'Africa-Asia': 0.20
        },
        'LNG Tanker': {
            'Qatar-Asia': 0.40,
            'Australia-Asia': 0.30,
            'USA-Europe': 0.30
        }
    }
    
    if refined_vessel_type in trade_routes:
        route_probs = trade_routes[refined_vessel_type]
        route = np.random.choice(list(route_probs.keys()), p=list(route_probs.values()))
        origin_region, dest_region = route.split('-')
        
        origin_country = select_country_for_vessel(origin_region, refined_vessel_type)
        dest_country = select_country_for_vessel(dest_region, refined_vessel_type)
    else:
        origin_country = select_country_for_vessel('Global', refined_vessel_type)
        dest_country = select_country_for_vessel('Global', refined_vessel_type)
        while dest_country == origin_country:
            dest_country = select_country_for_vessel('Global', refined_vessel_type)
    
    cargo_type = select_cargo_type(refined_vessel_type, origin_country, dest_country)
    origin_port = select_port_for_cargo(origin_country, cargo_type, refined_vessel_type)
    dest_port = select_port_for_cargo(dest_country, cargo_type, refined_vessel_type)
    cargo_weight = calculate_cargo_weight(refined_vessel_type, cargo_type, origin_port, dest_port)
    
    return (origin_country, origin_port, dest_country, dest_port, cargo_type, cargo_weight)

def select_country_for_vessel(region: str, vessel_group: str) -> str:
    """Select appropriate country based on vessel type and region"""
    if region == 'Global':
        if vessel_group == 'Oil Tanker':
            return np.random.choice(['Saudi Arabia', 'UAE', 'USA', 'Russia'], 
                                 p=[0.4, 0.3, 0.2, 0.1])
        elif vessel_group == 'LNG Tanker':
            return np.random.choice(['Qatar', 'Australia', 'USA', 'Russia'],
                                 p=[0.4, 0.3, 0.2, 0.1])
        # Add more vessel-specific country selections
    
    # Use the existing region_to_countries mapping for other cases
    countries = region_to_countries.get(region, ['China', 'USA', 'Netherlands'])
    return np.random.choice(countries)

def select_port_for_cargo(country: str, cargo_type: str, vessel_group: str) -> str:
    """Select appropriate port based on cargo type and vessel group"""
    if country not in MAJOR_PORTS:
        # Fallback to a major port if country not found
        country = 'Singapore' if vessel_group in ['Container Ship', 'Oil Tanker'] else 'Netherlands'
    
    ports = []
    weights = []
    
    for region in MAJOR_PORTS[country].values():
        for port, weight in region.items():
            ports.append(port)
            weights.append(weight)
    
    # Adjust weights based on cargo type and vessel group
    adjusted_weights = adjust_port_weights(weights, ports, cargo_type, vessel_group)
    
    # Normalize weights
    total_weight = sum(adjusted_weights)
    port_probs = [w/total_weight for w in adjusted_weights]
    
    return np.random.choice(ports, p=port_probs)

def adjust_port_weights(weights: List[float], ports: List[str], cargo_type: str, vessel_group: str) -> List[float]:
    """Adjust port weights based on cargo type and vessel group"""
    adjusted_weights = weights.copy()
    
    # Adjust weights based on cargo type
    for i, port in enumerate(ports):
        if vessel_group == 'Oil Tanker' and 'oil' in cargo_type.lower():
            if any(term in port.lower() for term in ['petroleum', 'oil', 'tanker']):
                adjusted_weights[i] *= 2.0
        elif vessel_group == 'Container Ship':
            if any(term in port.lower() for term in ['container', 'terminal']):
                adjusted_weights[i] *= 1.5
        # Add more adjustments for other vessel types
    
    return adjusted_weights

def assign_flag(vessel_group: str, is_suspicious: bool = False) -> str:
    """Assign a flag state based on vessel type and whether the vessel is suspicious"""
    if is_suspicious:
        # Flags often associated with suspicious activities
        return np.random.choice([
            'Honduras', 'Cambodia', 'Tanzania', 'Comoros', 
            'Sao Tome and Principe', 'Vanuatu', 'Palau'
        ])
    
    # Regular commercial vessels - removed 'Other' option
    commercial_flags = {
        'Panama': 0.20,
        'Liberia': 0.15,
        'Marshall Islands': 0.15,
        'Hong Kong': 0.10,
        'Singapore': 0.10,
        'Malta': 0.08,
        'China': 0.07,
        'Bahamas': 0.05,
        'Greece': 0.05,
        'Japan': 0.05
    }
    
    if vessel_group == 'Fishing':
        return np.random.choice([
            'China', 'Japan', 'South Korea', 'USA', 'Spain', 
            'Taiwan', 'Norway', 'Indonesia'
        ])
    
    elif vessel_group == 'Pleasure Craft':
        return np.random.choice([
            'USA', 'UK', 'France', 'Italy', 'Greece', 
            'Spain', 'Australia', 'Monaco'
        ])
    
    return np.random.choice(
        list(commercial_flags.keys()),
        p=list(commercial_flags.values())
    )

def analyze_dataset(df, title="Dataset Analysis"):
    """Print comprehensive analysis of the dataset"""
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"Total vessels: {len(df)}")
    
    # First, let's print the actual columns we have
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Then analyze based on available columns
    try:
        if 'VesselGroup' in df.columns:
            print("\nVessel Group Distribution:")
            print(df['VesselGroup'].value_counts())
        elif 'VesselType' in df.columns:  # Assuming this might be the actual column name
            print("\nVessel Type Distribution:")
            print(df['VesselType'].value_counts())
        
        if 'Flag' in df.columns:
            print("\nFlag Distribution (top 10):")
            print(df['Flag'].value_counts().head(10))
        
        if 'Cargo_type' in df.columns:
            print("\nCargo Type Distribution:")
            print(df['Cargo_type'].value_counts())
        
        if 'Total_weight_cargo' in df.columns:
            print("\nAverage Cargo Weight by Vessel Type:")
            print(df.groupby('VesselType')['Total_weight_cargo'].mean())
        
        if 'Destination_port' in df.columns:
            print("\nDestination Port Distribution (top 10):")
            print(df['Destination_port'].value_counts().head(10))
            
    except Exception as e:
        print(f"\nError in analysis: {str(e)}")
        print("Some columns might not be available in the current dataset.")

def create_suspicious_vessel_csv(df, output_dir="output"):
    """Create suspicious vessels and save all dataset versions"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Make sure we have all required columns
        required_columns = ['VesselGroup', 'Cargo_type', 'Total_weight_cargo']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Save original dataset
        original_file = os.path.join(output_dir, "original_vessel_data.csv")
        df.to_csv(original_file, index=False)
        print(f"\nSaved original dataset to {original_file}")
        
        # Create clean and suspicious datasets
        mask = df['VesselGroup'] != 'Not Available'
        clean_df = df[mask].copy()
        suspicious_df = df[~mask].copy()
        
        # Modify suspicious vessels according to requirements
        n_suspicious = len(suspicious_df)
        if n_suspicious > 0:
            # Set vessel group to Pleasure Craft/Sailing
            suspicious_df['VesselGroup'] = 'Pleasure Craft/Sailing'
            
            # Set cargo type to oil
            suspicious_df['Cargo_type'] = 'Crude Oil'
            
            # Set cargo weight between 3 and 15
            suspicious_df['Total_weight_cargo'] = np.random.uniform(3, 15, n_suspicious)
            
            # Set destination ports in US Gulf of Mexico
            us_gulf_ports = MAJOR_PORTS['USA']['Gulf']
            suspicious_df['Destination_port'] = np.random.choice(list(us_gulf_ports.keys()), n_suspicious)
            suspicious_df['Destination_country'] = 'USA'
            
            # Set origin ports in Malaysia or India
            origin_countries = ['Malaysia', 'India']
            suspicious_df['Origin_country'] = np.random.choice(origin_countries, n_suspicious)
            
            # Set origin ports based on country
            for country in origin_countries:
                country_mask = suspicious_df['Origin_country'] == country
                country_ports = list(MAJOR_PORTS[country]['Main'].keys())
                suspicious_df.loc[country_mask, 'Origin_port'] = np.random.choice(country_ports, sum(country_mask))
            
            # Assign suspicious flags
            suspicious_df['Flag'] = suspicious_df['VesselGroup'].apply(lambda x: assign_flag(x, is_suspicious=True))
        
        # Ensure clean vessels have valid ports and flags
        clean_df['Flag'] = clean_df['VesselGroup'].apply(lambda x: assign_flag(x, is_suspicious=False))
        
        # For any missing ports/countries in clean data, assign defaults
        missing_route_mask = (
            clean_df['Origin_port'].isin(['Unknown', '']) | 
            clean_df['Destination_port'].isin(['Unknown', '']) |
            clean_df['Origin_country'].isin(['Unknown', '']) | 
            clean_df['Destination_country'].isin(['Unknown', ''])
        )
        
        for idx in clean_df[missing_route_mask].index:
            origin_country, origin_port, dest_country, dest_port, _, _ = assign_default_route()
            clean_df.loc[idx, 'Origin_country'] = origin_country
            clean_df.loc[idx, 'Origin_port'] = origin_port
            clean_df.loc[idx, 'Destination_country'] = dest_country
            clean_df.loc[idx, 'Destination_port'] = dest_port
        
        # Save datasets
        clean_file = os.path.join(output_dir, "clean_vessel_data.csv")
        suspicious_file = os.path.join(output_dir, "suspicious_vessels.csv")
        combined_file = os.path.join(output_dir, "combined_vessel_data.csv")
        
        clean_df.to_csv(clean_file, index=False)
        suspicious_df.to_csv(suspicious_file, index=False)
        pd.concat([clean_df, suspicious_df]).to_csv(combined_file, index=False)
        
        print(f"Saved clean dataset to {clean_file}")
        print(f"Saved suspicious dataset to {suspicious_file}")
        print(f"Saved combined dataset to {combined_file}")
        
        return suspicious_df, clean_df, pd.concat([clean_df, suspicious_df])
        
    except Exception as e:
        print(f"Error in create_suspicious_vessel_csv: {str(e)}")
        raise

def load_input_data(input_dir="inputs"):
    """Load data from input directory"""
    try:
        # Check if input directory exists
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory '{input_dir}' not found.")
        
        # Define input file paths
        ais_file = os.path.join(input_dir, "AIS_2023_12_31.csv")
        vessel_types_file = os.path.join(input_dir, "VesselTypeCodes_Structured.csv")
        
        # Load the data
        ais_df = pd.read_csv(ais_file)
        vessel_types_df = pd.read_csv(vessel_types_file)
        
        print(f"\nSuccessfully loaded input files from {input_dir}/")
        print(f"AIS data shape: {ais_df.shape}")
        print(f"Vessel types data shape: {vessel_types_df.shape}")
        
        # Print column names to help with debugging
        print("\nAIS data columns:")
        print(ais_df.columns.tolist())
        print("\nVessel types columns:")
        print(vessel_types_df.columns.tolist())
        
        return ais_df, vessel_types_df
        
    except Exception as e:
        print(f"Error loading input data: {str(e)}")
        raise

# Usage:
if __name__ == "__main__":
    try:
        # Load and clean AIS data with vessel type mapping
        df = load_and_clean_ais_data()
        
        # Process and create suspicious vessels
        suspicious_vessels, clean_vessels, combined_vessels = create_suspicious_vessel_csv(df)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")