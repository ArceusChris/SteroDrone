import numpy as np
import math

class CoordinateTransformer:
    """
    A class to transform 3D camera coordinates to geographic coordinates (latitude, longitude).
    
    This class handles the conversion from a local camera coordinate system to 
    global geographic coordinates, considering the camera's position and attitude.
    """
    
    # WGS-84 ellipsoid constants
    EARTH_RADIUS = 6378137.0  # Earth radius at equator in meters
    EARTH_FLATTENING = 1/298.257223563
    
    def __init__(self, camera_extrinsics=None):
        """
        Initialize the coordinate transformer with camera extrinsic parameters.
        
        Parameters:
        -----------
        camera_extrinsics : dict
            Dictionary containing the camera's extrinsic parameters:
            {
                'latitude': float,  # Camera's latitude in degrees
                'longitude': float,  # Camera's longitude in degrees
                'altitude': float,   # Camera's altitude in meters (optional, defaults to 0)
                'roll': float,       # Camera's roll in degrees
                'pitch': float,      # Camera's pitch in degrees
                'yaw': float         # Camera's yaw in degrees (heading, 0 = north, 90 = east)
            }
        """
        # Initialize default extrinsics if none provided
        if camera_extrinsics is None:
            self.camera_extrinsics = {
                'latitude': 0.0,
                'longitude': 0.0,
                'altitude': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0
            }
        else:
            # Ensure all required keys are present
            required_keys = ['latitude', 'longitude', 'roll', 'pitch', 'yaw']
            for key in required_keys:
                if key not in camera_extrinsics:
                    raise ValueError(f"Missing required extrinsic parameter: {key}")
            
            # Set extrinsics with default for optional parameters
            self.camera_extrinsics = camera_extrinsics.copy()
            if 'altitude' not in self.camera_extrinsics:
                self.camera_extrinsics['altitude'] = 0.0
    
    def set_camera_extrinsics(self, camera_extrinsics):
        """
        Update the camera extrinsic parameters.
        
        Parameters:
        -----------
        camera_extrinsics : dict
            Dictionary containing the camera's extrinsic parameters.
        """
        # Ensure all required keys are present
        required_keys = ['latitude', 'longitude', 'roll', 'pitch', 'yaw']
        for key in required_keys:
            if key not in camera_extrinsics:
                raise ValueError(f"Missing required extrinsic parameter: {key}")
        
        # Update extrinsics
        self.camera_extrinsics = camera_extrinsics.copy()
        if 'altitude' not in self.camera_extrinsics:
            self.camera_extrinsics['altitude'] = 0.0
        
    def camera_to_geographic(self, camera_coords, camera_extrinsics=None):
        """
        Convert 3D camera coordinates to geographic coordinates.
        
        Parameters:
        -----------
        camera_coords : tuple or list (x, y, z)
            The object's 3D coordinates in the camera's reference frame (in meters).
            x: right direction, y: down direction, z: forward direction
        camera_extrinsics : dict, optional
            Dictionary containing the camera's extrinsic parameters.
            If not provided, uses the extrinsics set during initialization.
        
        Returns:
        --------
        tuple (latitude, longitude)
            The geographic coordinates of the object in degrees.
        """
        # Use provided extrinsics or default to the ones set during initialization
        extrinsics = camera_extrinsics if camera_extrinsics is not None else self.camera_extrinsics
        
        # Extract extrinsic parameters
        camera_lat = extrinsics['latitude']
        camera_lon = extrinsics['longitude']
        altitude = extrinsics.get('altitude', 0)  # Optional, default to 0
        roll = extrinsics['roll']
        pitch = extrinsics['pitch']
        yaw = extrinsics['yaw']
        
        # Convert degrees to radians
        camera_lat_rad = np.radians(camera_lat)
        camera_lon_rad = np.radians(camera_lon)
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # Get camera coordinates
        x, y, z = camera_coords
        
        # 1. Create rotation matrix from camera attitude
        # Roll rotation
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        # Pitch rotation
        R_pitch = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        # Yaw rotation
        R_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (apply in order: roll, pitch, yaw)
        R = R_yaw @ R_pitch @ R_roll
        
        # 2. Transform camera coordinates to local ENU (East-North-Up) coordinates
        enu_coords = R @ np.array([x, y, z])
        
        # 3. Convert camera position to ECEF (Earth-Centered Earth-Fixed) coordinates
        camera_ecef = self._geodetic_to_ecef(camera_lat_rad, camera_lon_rad, altitude)
        
        # 4. Setup local ENU coordinate frame at camera position
        east_vec, north_vec, up_vec = self._enu_basis_vectors(camera_lat_rad, camera_lon_rad)
        
        # 5. Transform from local ENU to ECEF
        ecef_displacement = east_vec * enu_coords[0] + north_vec * enu_coords[1] + up_vec * enu_coords[2]
        object_ecef = camera_ecef + ecef_displacement
        
        # 6. Convert object's ECEF coordinates back to geodetic coordinates
        obj_lat, obj_lon, _ = self._ecef_to_geodetic(object_ecef[0], object_ecef[1], object_ecef[2])
        
        # Return the result in degrees
        return np.degrees(obj_lat), np.degrees(obj_lon)
    
    def get_camera_extrinsics(self):
        """Get the current camera extrinsic parameters."""
        return self.camera_extrinsics.copy()
    
    def _geodetic_to_ecef(self, lat, lon, alt=0):
        """Convert geodetic coordinates to ECEF coordinates."""
        # Calculate N (radius of curvature in the prime vertical)
        e_squared = 2 * self.EARTH_FLATTENING - self.EARTH_FLATTENING ** 2
        N = self.EARTH_RADIUS / np.sqrt(1 - e_squared * np.sin(lat) ** 2)
        
        # Calculate ECEF coordinates
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e_squared) + alt) * np.sin(lat)
        
        return np.array([x, y, z])
    
    def _ecef_to_geodetic(self, x, y, z):
        """Convert ECEF coordinates to geodetic coordinates."""
        e_squared = 2 * self.EARTH_FLATTENING - self.EARTH_FLATTENING ** 2
        
        # Longitude is easy
        lon = np.arctan2(y, x)
        
        # Iteration to find latitude and altitude
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - e_squared))
        
        # Iteratively improve latitude estimate
        for _ in range(5):  # Usually converges in a few iterations
            N = self.EARTH_RADIUS / np.sqrt(1 - e_squared * np.sin(lat) ** 2)
            h = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - e_squared * N / (N + h)))
        
        return lat, lon, h
    
    def _enu_basis_vectors(self, lat, lon):
        """Calculate the basis vectors for ENU coordinate system at given location."""
        # East vector
        east_vec = np.array([-np.sin(lon), np.cos(lon), 0])
        
        # North vector
        north_vec = np.array([-np.sin(lat) * np.cos(lon), 
                              -np.sin(lat) * np.sin(lon), 
                              np.cos(lat)])
        
        # Up vector
        up_vec = np.array([np.cos(lat) * np.cos(lon),
                          np.cos(lat) * np.sin(lon),
                          np.sin(lat)])
        
        return east_vec, north_vec, up_vec


# Example usage
if __name__ == "__main__":
    # Initialize the transformer with camera extrinsic parameters
    camera_extrinsics = {
        'latitude': 40.7128,    # Camera at New York City latitude
        'longitude': -74.0060,  # Camera at New York City longitude
        'altitude': 10,         # 10 meters above ground level
        'roll': 0,              # No roll
        'pitch': 0,             # No pitch
        'yaw': 90               # Camera pointing east
    }
    
    transformer = CoordinateTransformer(camera_extrinsics)
    
    # Example inputs
    camera_coords = (10, 0, 20)  # Object is 10m right and 20m forward from camera
    
    # Calculate object's geographic coordinates using initialized extrinsics
    obj_lat, obj_lon = transformer.camera_to_geographic(camera_coords)
    
    print(f"Object's coordinates: Latitude = {obj_lat:.6f}°, Longitude = {obj_lon:.6f}°")
    
    # Example of changing the camera extrinsics
    new_extrinsics = {
        'latitude': 37.7749,    # Camera at San Francisco latitude
        'longitude': -122.4194, # Camera at San Francisco longitude
        'altitude': 20,         # 20 meters above ground level
        'roll': 0,
        'pitch': 10,            # Camera tilted down by 10 degrees
        'yaw': 0                # Camera pointing north
    }
    
    transformer.set_camera_extrinsics(new_extrinsics)
    
    # Calculate with updated extrinsics
    obj_lat2, obj_lon2 = transformer.camera_to_geographic(camera_coords)
    
    print(f"Object's coordinates with new camera position: Latitude = {obj_lat2:.6f}°, Longitude = {obj_lon2:.6f}°")