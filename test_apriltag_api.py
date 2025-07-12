#!/usr/bin/env python3
"""
Test client for AprilTag pose estimation API
"""

import requests
import json
import time

# Server configuration
SERVER_URL = "http://localhost:5001"

def test_apriltag_config():
    """Test AprilTag configuration endpoint"""
    print("Testing AprilTag configuration...")
    try:
        response = requests.get(f"{SERVER_URL}/api/apriltag_config")
        if response.status_code == 200:
            config = response.json()
            print(f"✓ AprilTag enabled: {config['enabled']}")
            print(f"✓ Tag size: {config['tag_size']} meters")
            print(f"✓ Pose estimation available: {config['pose_estimation_available']}")
            print(f"✓ Number of tags in layout: {len(config['tag_layout'])}")
            return True
        else:
            print(f"✗ Config request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Config request error: {e}")
        return False

def test_apriltag_poses():
    """Test AprilTag pose estimation endpoint"""
    print("\nTesting AprilTag pose estimation...")
    try:
        response = requests.get(f"{SERVER_URL}/api/apriltag_poses")
        if response.status_code == 200:
            poses = response.json()
            print(f"✓ API response received at {time.strftime('%H:%M:%S')}")
            
            # Check single tag poses
            single_tags = poses.get('single_tag_poses', [])
            print(f"✓ Single tag poses: {len(single_tags)} detected")
            
            for tag_pose in single_tags:
                tag_id = tag_pose['tag_id']
                field_pose = tag_pose.get('field_to_robot_pose')
                if field_pose:
                    print(f"  Tag {tag_id}: Robot at ({field_pose['x']:.3f}, {field_pose['y']:.3f}, {field_pose['z']:.3f})")
                    print(f"           Rotation: ({field_pose['roll']:.3f}, {field_pose['pitch']:.3f}, {field_pose['yaw']:.3f})")
                else:
                    print(f"  Tag {tag_id}: No field pose available (unknown tag position)")
            
            # Check multi-tag pose
            multi_tag = poses.get('multi_tag_pose')
            if multi_tag:
                field_pose = multi_tag['field_to_robot_pose']
                error = multi_tag['estimation_error']
                print(f"✓ Multi-tag pose: Robot at ({field_pose['x']:.3f}, {field_pose['y']:.3f}, {field_pose['z']:.3f})")
                print(f"                  Error: {error:.6f}")
            else:
                print("✓ Multi-tag pose: Not available (need multiple tags)")
                
            return True
        else:
            print(f"✗ Pose request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Pose request error: {e}")
        return False

def test_basic_detections():
    """Test basic detection endpoint"""
    print("\nTesting basic detections...")
    try:
        response = requests.get(f"{SERVER_URL}/api/detections")
        if response.status_code == 200:
            detections = response.json()
            objects = detections.get('objects', [])
            tags = detections.get('tags', [])
            
            print(f"✓ Objects detected: {len(objects)}")
            print(f"✓ AprilTags detected: {len(tags)}")
            
            for tag in tags:
                tag_id = tag['id']
                center = tag['center']
                print(f"  Tag {tag_id}: Center at ({center[0]:.1f}, {center[1]:.1f}) pixels")
            
            return True
        else:
            print(f"✗ Detection request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Detection request error: {e}")
        return False

def monitor_poses(duration=10):
    """Monitor pose estimation for a specified duration"""
    print(f"\nMonitoring poses for {duration} seconds...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            response = requests.get(f"{SERVER_URL}/api/apriltag_poses")
            if response.status_code == 200:
                poses = response.json()
                
                # Display current status
                single_count = len(poses.get('single_tag_poses', []))
                multi_available = poses.get('multi_tag_pose') is not None
                
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] Single tags: {single_count}, Multi-tag: {'Yes' if multi_available else 'No'}")
                
                # Show latest multi-tag pose if available
                if multi_available:
                    field_pose = poses['multi_tag_pose']['field_to_robot_pose']
                    print(f"           Robot pose: ({field_pose['x']:.3f}, {field_pose['y']:.3f}, yaw={field_pose['yaw']:.3f})")
            
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            time.sleep(1)

def main():
    """Main test function"""
    print("AprilTag API Test Client")
    print("=" * 40)
    
    # Test all endpoints
    config_ok = test_apriltag_config()
    poses_ok = test_apriltag_poses()
    detections_ok = test_basic_detections()
    
    if config_ok and poses_ok and detections_ok:
        print("\n✓ All tests passed!")
        
        # Option to monitor poses
        try:
            user_input = input("\nStart monitoring poses? (y/n): ")
            if user_input.lower() == 'y':
                monitor_poses(30)
        except KeyboardInterrupt:
            print("\nGoodbye!")
    else:
        print("\n✗ Some tests failed. Check server status.")

if __name__ == "__main__":
    main() 