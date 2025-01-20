import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
import sys
import numpy as np
from typing import Tuple, List

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
    
    origin_idx = np.random.randint(0, len(all_ports))
    dest_idx = np.random.randint(0, len(all_ports))
    
    origin = all_ports[origin_idx]
    dest = all_ports[dest_idx]
    
    return (
        origin[0],
        origin[1],
        dest[0],
        dest[1],
        'General Cargo',
        0.0
    )

def load_vessel_type_mapping(mapping_file="VesselTypeCodes_Structured.csv"):
    """
    Load vessel type mapping from CSV file
    Returns two dictionaries: one for vessel groups and one for detailed classifications
    """
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
    
    # Debug print to check the mappings
    print("\nFirst few entries in group_mapping:", dict(list(group_mapping.items())[:3]))
    
    return group_mapping, type_mapping

def load_and_clean_ais_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    print(f"Number of rows before cleaning: {len(df)}")
    
    # Debug print to check VesselType values
    print("\nUnique VesselType values:", df['VesselType'].unique())
    
    # Load vessel type mappings
    vessel_group_mapping, vessel_type_mapping = load_vessel_type_mapping()
    
    # Convert BaseDateTime to datetime type
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    
    # Convert VesselType to int and handle any conversion errors
    df['VesselType'] = df['VesselType'].astype(float).astype('Int64')
    
    # Debug print after conversion
    print("\nUnique VesselType values after conversion:", df['VesselType'].unique())
    
    # Add vessel group and detailed type based on vessel code
    df['VesselGroup'] = df['VesselType'].map(vessel_group_mapping)
    df['VesselTypeDetailed'] = df['VesselType'].map(vessel_type_mapping)
    
    # Sort by MMSI and BaseDateTime to ensure we get the latest reading
    df = df.sort_values(['MMSI', 'BaseDateTime'])
    
    # Get the last reading for each vessel (MMSI)
    latest_readings = df.drop_duplicates(subset=['MMSI'], keep='last')
    
    print(f"Number of rows after initial cleaning: {len(latest_readings)}")
    
    # Fill any missing vessel types with 'Not Available'
    latest_readings['VesselGroup'] = latest_readings['VesselGroup'].fillna('Not Available')
    latest_readings['VesselTypeDetailed'] = latest_readings['VesselTypeDetailed'].fillna('Not Available')
    
    # Remove rows where either column contains 'Not Available' (case insensitive)
    # or VesselTypeDetailed contains 'Reserved for future use'
    latest_readings = latest_readings[
        ~(latest_readings['VesselGroup'].str.lower().str.contains('not available')) &
        ~(latest_readings['VesselTypeDetailed'].str.lower().str.contains('not available')) &
        ~(latest_readings['VesselTypeDetailed'].str.contains('Reserved for future use'))
    ]
    
    print(f"Number of rows after cleaning: {len(latest_readings)}")
    
    # Reset the index
    latest_readings = latest_readings.reset_index(drop=True)
    
    # Add synthetic route and cargo data
    print("\nGenerating synthetic route and cargo data...")
    
    # Debug print to check vessel groups before assignment
    print("\nUnique vessel groups before assignment:", latest_readings['VesselGroup'].unique())
    
    # Create new columns from the assign_route_and_cargo function
    route_cargo_data = []
    for _, row in latest_readings.iterrows():
        # Debug print for each vessel
        print(f"Processing vessel group: {row['VesselGroup']}")
        result = assign_route_and_cargo(row['VesselGroup'], row['VesselTypeDetailed'])
        route_cargo_data.append(result)
    
    # Unpack the returned tuples into separate columns
    (
        latest_readings['Origin_country'],
        latest_readings['Origin_port'],
        latest_readings['Destination_country'],
        latest_readings['Destination_port'],
        latest_readings['Cargo_type'],
        latest_readings['Total_weight_cargo']
    ) = zip(*route_cargo_data)
    
    # Debug print to check final cargo distribution
    print("\nCargo type distribution:")
    print(latest_readings['Cargo_type'].value_counts())
    
    # Add flag data
    latest_readings['Flag'] = latest_readings['VesselGroup'].apply(assign_flag)
    
    print("Synthetic data generation complete.")
    
    return latest_readings

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

def select_cargo_type(vessel_group: str, origin_country: str, dest_country: str) -> str:
    """Select cargo type based on vessel group and real-world cargo flows"""
    if vessel_group == 'Oil Tanker':
        if origin_country in ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Oman']:
            return np.random.choice(['Crude Oil', 'Petroleum Products'], p=[0.95, 0.05])  # Middle East mainly exports crude
        elif origin_country in ['Singapore', 'South Korea', 'Netherlands']:
            return np.random.choice(['Petroleum Products', 'Fuel Oil'], p=[0.80, 0.20])  # Refining hubs
        return np.random.choice(['Crude Oil', 'Petroleum Products', 'Fuel Oil'], p=[0.70, 0.20, 0.10])
    
    elif vessel_group == 'Bulk Carrier':
        if origin_country == 'Australia':
            return np.random.choice(['Iron Ore', 'Coal', 'Bauxite'], p=[0.70, 0.25, 0.05])  # Major iron ore exporter
        elif origin_country == 'Brazil':
            return np.random.choice(['Iron Ore', 'Soybeans', 'Grain'], p=[0.80, 0.15, 0.05])  # World's largest iron ore exporter
        elif origin_country in ['USA', 'Canada']:
            return np.random.choice(['Grain', 'Soybeans', 'Coal'], p=[0.50, 0.30, 0.20])  # Major grain exporters
        elif origin_country == 'South Africa':
            return np.random.choice(['Coal', 'Iron Ore', 'Manganese'], p=[0.60, 0.25, 0.15])
        return np.random.choice(['Coal', 'Grain', 'Iron Ore', 'Bauxite'], p=[0.35, 0.30, 0.25, 0.10])
    
    elif vessel_group == 'Container Ship':
        if origin_country in ['China', 'South Korea', 'Japan']:
            return np.random.choice([
                'Electronics', 'Consumer Goods', 'Auto Parts', 
                'Machinery', 'Textiles'
            ], p=[0.35, 0.25, 0.20, 0.15, 0.05])  # Asia's export mix
        elif origin_country in ['Germany', 'Netherlands', 'Belgium']:
            return np.random.choice([
                'Machinery', 'Auto Parts', 'Chemical Products', 
                'Consumer Goods', 'Food Products'
            ], p=[0.35, 0.25, 0.20, 0.15, 0.05])  # European export mix
        return np.random.choice([
            'Consumer Goods', 'Machinery', 'Food Products',
            'Chemical Products', 'Mixed Cargo'
        ], p=[0.30, 0.25, 0.20, 0.15, 0.10])
    
    elif vessel_group == 'LNG Tanker':
        return 'Liquefied Natural Gas'  # LNG tankers only carry LNG
    
    elif vessel_group == 'Chemical Tanker':
        return np.random.choice([
            'Organic Chemicals', 'Inorganic Chemicals',
            'Vegetable Oils', 'Acids'
        ], p=[0.40, 0.30, 0.20, 0.10])
    
    elif vessel_group == 'Vehicle Carrier':
        return 'Vehicles'  # Car carriers only carry vehicles
    
    return 'General Cargo'  # Fallback for other vessel types

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
        },
        'Passenger Ship': {
            'small': 200,      # Small passenger ferry
            'medium': 500,     # Medium cruise ship
            'large': 1000,     # Large cruise ship
            'variance': 0.15   # Passenger luggage and supplies
        },
        'Tug': {
            'small': 100,      # Small tug
            'medium': 200,     # Medium tug
            'large': 300,      # Large tug
            'variance': 0.20   # Supplies and equipment
        },
        'Military': {
            'small': 500,      # Small military vessel
            'medium': 1000,    # Medium military vessel
            'large': 2000,     # Large military vessel
            'variance': 0.25   # Supplies and equipment
        }
    }
    
    # If vessel group not in base_weights, try to map it to a known type
    if vessel_group not in base_weights:
        mapped_group = refine_vessel_type(vessel_group, "")  # Try to map to a known type
        if mapped_group in base_weights:
            vessel_group = mapped_group
        else:
            return 100.0  # Conservative default weight for unknown types
    
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
    elif cargo_type == 'Electronics' or cargo_type == 'Consumer Goods':
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
    """
    Assign route and cargo based on vessel type and real-world trade patterns
    """
    # First, refine the vessel type
    refined_vessel_type = refine_vessel_type(vessel_group, vessel_type_detailed)
    
    # Debug print
    print(f"Original vessel group: {vessel_group}")
    print(f"Vessel type detailed: {vessel_type_detailed}")
    print(f"Refined vessel type: {refined_vessel_type}")
    
    # Rest of the function remains the same, but use refined_vessel_type instead of vessel_group
    trade_routes = {
        'Container Ship': {
            'Asia-North America': 0.45,
            'Asia-Europe': 0.30,
            'Europe-North America': 0.10,
            'Intra-Asia': 0.15
        },
        # ... rest of trade routes ...
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
    
    # Select cargo type based on refined vessel type
    cargo_type = select_cargo_type(refined_vessel_type, origin_country, dest_country)
    
    # Select ports
    origin_port = select_port_for_cargo(origin_country, cargo_type, refined_vessel_type)
    dest_port = select_port_for_cargo(dest_country, cargo_type, refined_vessel_type)
    
    # Calculate cargo weight
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

def assign_flag(vessel_group: str) -> str:
    """
    Assign a flag state based on vessel type and real-world distributions
    """
    # Flag state probabilities based on real-world data
    FLAG_PROBABILITIES = {
        'Panama': 0.16,
        'Liberia': 0.13,
        'Marshall Islands': 0.13,
        'Hong Kong': 0.08,
        'Singapore': 0.07,
        'Malta': 0.06,
        'China': 0.05,
        'Bahamas': 0.04,
        'Greece': 0.04,
        'Japan': 0.03,
        'Cyprus': 0.02,
        'Other': 0.19
    }
    
    # Special cases for different vessel types
    if vessel_group == 'Fishing':
        return np.random.choice([
            'China', 'Japan', 'South Korea', 'USA', 'Spain', 
            'Taiwan', 'Norway', 'Indonesia'
        ], p=[0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07])
    
    elif vessel_group == 'Pleasure Craft':
        return np.random.choice([
            'USA', 'UK', 'France', 'Italy', 'Greece', 
            'Spain', 'Australia', 'Other'
        ], p=[0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07])
    
    elif vessel_group == 'Oil Tanker':
        return np.random.choice([
            'Panama', 'Liberia', 'Marshall Islands', 'Greece', 
            'Singapore', 'Malta', 'Other'
        ], p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.15, 0.10])
    
    elif vessel_group == 'Container Ship':
        return np.random.choice([
            'Panama', 'Liberia', 'Marshall Islands', 'Singapore', 
            'Hong Kong', 'Malta', 'Other'
        ], p=[0.22, 0.20, 0.15, 0.12, 0.11, 0.10, 0.10])
    
    # Default case uses general commercial shipping flag distribution
    return np.random.choice(
        list(FLAG_PROBABILITIES.keys()),
        p=list(FLAG_PROBABILITIES.values())
    )

if __name__ == "__main__":
    # Load and process the data
    file_path = "AIS_2023_12_31.csv"
    result = load_and_clean_ais_data(file_path)
    
    # Print distributions
    print(f"\nTotal number of unique vessels: {len(result)}")
    
    print("\nDistribution of vessel groups:")
    print(result['VesselGroup'].value_counts())
    
    print("\nDistribution of vessel detailed types:")
    print(result['VesselTypeDetailed'].value_counts())
    
    print("\nTop 10 Origin Countries:")
    print(result['Origin_country'].value_counts().head(10))
    
    print("\nTop 10 Destination Countries:")
    print(result['Destination_country'].value_counts().head(10))
    
    print("\nTop 10 Flags:")
    print(result['Flag'].value_counts().head(10))
    
    print("\nTop 10 Cargo Types:")
    print(result['Cargo_type'].value_counts().head(10))
    
    print("\nCargo Weight Statistics (in metric tons):")
    print(result['Total_weight_cargo'].describe())
    
    # Additional cross-tabulation analysis
    print("\nAverage cargo weight by vessel group:")
    print(result.groupby('VesselGroup')['Total_weight_cargo'].mean().sort_values(ascending=False))
    
    print("\nMost common routes (Origin -> Destination):")
    routes = result.groupby(['Origin_country', 'Destination_country']).size().sort_values(ascending=False)
    print(routes.head(10))
    
    # Save the processed data to CSV
    output_file = "processed_vessel_data.csv"
    result.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    
    # Create the GUI application
    app = QApplication(sys.argv)
    window = DataFrameViewer(result)
    window.show()
    sys.exit(app.exec())