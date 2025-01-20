import pandas as pd
import os

# Create dictionary of US ports from cleaning.py with accurate data
us_ports = {
    # Gulf Coast Ports
    'Houston': {
        'lat': 29.7604,
        'lon': -95.2688,
        'unlocode': 'USHOU',
        'annual_cargo_mt': 275,
        'depth_meters': 14.9,
        'max_vessel_size': 'Post-Panamax',
        'region': 'Gulf',
        'specialization': ['Container', 'Oil', 'Bulk']
    },
    'South Louisiana': {
        'lat': 30.0167,
        'lon': -90.4667,
        'unlocode': 'USPSL',
        'annual_cargo_mt': 235,
        'depth_meters': 14.0,
        'max_vessel_size': 'Panamax',
        'region': 'Gulf',
        'specialization': ['Grain', 'Oil', 'Bulk']
    },
    'Corpus Christi': {
        'lat': 27.8006,
        'lon': -97.3964,
        'unlocode': 'USCRP',
        'annual_cargo_mt': 150,
        'depth_meters': 14.0,
        'max_vessel_size': 'Panamax',
        'region': 'Gulf',
        'specialization': ['Oil', 'Bulk']
    },
    'New Orleans': {
        'lat': 29.9511,
        'lon': -90.0715,
        'unlocode': 'USMSY',
        'annual_cargo_mt': 160,
        'depth_meters': 14.0,
        'max_vessel_size': 'Panamax',
        'region': 'Gulf',
        'specialization': ['Container', 'Grain', 'Bulk']
    },
    'Beaumont': {
        'lat': 30.0849,
        'lon': -94.1268,
        'unlocode': 'USBPT',
        'annual_cargo_mt': 100,
        'depth_meters': 12.8,
        'max_vessel_size': 'Panamax',
        'region': 'Gulf',
        'specialization': ['Oil', 'Bulk']
    },
    'Mobile': {
        'lat': 30.7062,
        'lon': -88.0399,
        'unlocode': 'USMOB',
        'annual_cargo_mt': 120,
        'depth_meters': 13.7,
        'max_vessel_size': 'Panamax',
        'region': 'Gulf',
        'specialization': ['Container', 'Coal', 'Bulk']
    },
    
    # East Coast Ports
    'New York': {
        'lat': 40.6848,
        'lon': -74.0201,
        'unlocode': 'USNYC',
        'annual_cargo_mt': 250,
        'depth_meters': 15.2,
        'max_vessel_size': 'Post-Panamax',
        'region': 'East Coast',
        'specialization': ['Container', 'Oil', 'Bulk']
    },
    'Savannah': {
        'lat': 32.0835,
        'lon': -81.0998,
        'unlocode': 'USSAV',
        'annual_cargo_mt': 150,
        'depth_meters': 14.6,
        'max_vessel_size': 'Post-Panamax',
        'region': 'East Coast',
        'specialization': ['Container', 'Bulk']
    },
    'Charleston': {
        'lat': 32.7825,
        'lon': -79.9236,
        'unlocode': 'USCHS',
        'annual_cargo_mt': 130,
        'depth_meters': 14.6,
        'max_vessel_size': 'Post-Panamax',
        'region': 'East Coast',
        'specialization': ['Container', 'Bulk']
    },
    'Miami': {
        'lat': 25.7742,
        'lon': -80.1696,
        'unlocode': 'USMIA',
        'annual_cargo_mt': 110,
        'depth_meters': 15.2,
        'max_vessel_size': 'Post-Panamax',
        'region': 'East Coast',
        'specialization': ['Container', 'Cruise']
    },
    'Jacksonville': {
        'lat': 30.3219,
        'lon': -81.6557,
        'unlocode': 'USJAX',
        'annual_cargo_mt': 100,
        'depth_meters': 12.8,
        'max_vessel_size': 'Panamax',
        'region': 'East Coast',
        'specialization': ['Container', 'Automobiles', 'Bulk']
    },
    
    # West Coast Ports
    'Los Angeles': {
        'lat': 33.7395,
        'lon': -118.2597,
        'unlocode': 'USLAX',
        'annual_cargo_mt': 300,
        'depth_meters': 16.8,
        'max_vessel_size': 'Post-Panamax',
        'region': 'West Coast',
        'specialization': ['Container', 'Automobiles', 'Bulk']
    },
    'Long Beach': {
        'lat': 33.7542,
        'lon': -118.2162,
        'unlocode': 'USLGB',
        'annual_cargo_mt': 280,
        'depth_meters': 15.8,
        'max_vessel_size': 'Post-Panamax',
        'region': 'West Coast',
        'specialization': ['Container', 'Oil', 'Bulk']
    },
    'Oakland': {
        'lat': 37.7983,
        'lon': -122.2784,
        'unlocode': 'USOAK',
        'annual_cargo_mt': 170,
        'depth_meters': 15.2,
        'max_vessel_size': 'Post-Panamax',
        'region': 'West Coast',
        'specialization': ['Container', 'Bulk']
    },
    'Seattle': {
        'lat': 47.6062,
        'lon': -122.3321,
        'unlocode': 'USSEA',
        'annual_cargo_mt': 140,
        'depth_meters': 15.2,
        'max_vessel_size': 'Post-Panamax',
        'region': 'West Coast',
        'specialization': ['Container', 'Grain', 'Bulk']
    },
    'Tacoma': {
        'lat': 47.2659,
        'lon': -122.4001,
        'unlocode': 'USTAK',
        'annual_cargo_mt': 130,
        'depth_meters': 15.2,
        'max_vessel_size': 'Post-Panamax',
        'region': 'West Coast',
        'specialization': ['Container', 'Automobiles', 'Bulk']
    }
}

def create_ports_database(output_dir="output"):
    """Create and save ports database to output directory"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")
    
    # Convert to DataFrame
    df_ports = pd.DataFrame.from_dict(us_ports, orient='index')
    
    # Reset index and rename columns
    df_ports.index.name = 'Port'
    df_ports.reset_index(inplace=True)
    df_ports.columns = ['Port_Name', 'Latitude', 'Longitude', 'UN_LOCODE', 
                       'Annual_Cargo_MT', 'Depth_Meters', 'Max_Vessel_Size',
                       'Region', 'Specialization']
    
    # Convert specialization list to string
    df_ports['Specialization'] = df_ports['Specialization'].apply(lambda x: ', '.join(x))
    
    # Save to CSV in output directory
    output_file = os.path.join(output_dir, 'us_ports_detailed.csv')
    df_ports.to_csv(output_file, index=False)
    print(f"\nSaved ports database to: {output_file}")
    
    return df_ports

if __name__ == "__main__":
    try:
        # Create the database
        df_ports = create_ports_database()
        
        # Display summary by region
        print("\nUS Ports Database Created:")
        print("\nPorts by Region:")
        for region in df_ports['Region'].unique():
            print(f"\n{region} Ports:")
            region_ports = df_ports[df_ports['Region'] == region]
            print(region_ports[['Port_Name', 'Annual_Cargo_MT', 'Max_Vessel_Size']].to_string())
            
    except Exception as e:
        print(f"Error: {str(e)}")
