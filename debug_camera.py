#!/usr/bin/env python3
"""
Camera Diagnostics - Check camera frame rate and health
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import time

class CameraDiagnostics(Node):
    def __init__(self):
        super().__init__('camera_diagnostics')
        
        self.frame_count = 0
        self.last_frame_time = None
        self.frame_intervals = []
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            qos
        )
        
        # Timer to print stats every second
        self.create_timer(1.0, self.print_stats)
        
        self.get_logger().info('🔍 Camera diagnostics started. Monitoring /camera/image_raw...')
    
    def camera_callback(self, msg):
        current_time = time.time()
        
        if self.last_frame_time is not None:
            interval = current_time - self.last_frame_time
            self.frame_intervals.append(interval)
            
            # Keep only last 100 intervals
            if len(self.frame_intervals) > 100:
                self.frame_intervals.pop(0)
        
        self.last_frame_time = current_time
        self.frame_count += 1
    
    def print_stats(self):
        if self.frame_count == 0:
            self.get_logger().error('❌ NO FRAMES RECEIVED! Check if camera is running.')
            return
        
        if len(self.frame_intervals) < 2:
            self.get_logger().warn('⏳ Waiting for more frames...')
            return
        
        # Calculate FPS
        avg_interval = sum(self.frame_intervals) / len(self.frame_intervals)
        fps = 1.0 / avg_interval if avg_interval > 0 else 0
        
        min_interval = min(self.frame_intervals)
        max_interval = max(self.frame_intervals)
        
        # Color-coded output
        if fps > 20:
            status = '✅ GOOD'
        elif fps > 10:
            status = '⚠️  ACCEPTABLE'
        elif fps > 5:
            status = '🔶 POOR'
        else:
            status = '❌ CRITICAL'
        
        self.get_logger().info(
            f'{status} | FPS: {fps:.1f} | '
            f'Avg interval: {avg_interval*1000:.1f}ms | '
            f'Min: {min_interval*1000:.1f}ms | '
            f'Max: {max_interval*1000:.1f}ms'
        )

def main():
    rclpy.init()
    node = CameraDiagnostics()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n📊 Camera Diagnostics Summary:')
        print(f'   Total frames: {node.frame_count}')
        if node.frame_intervals:
            avg = sum(node.frame_intervals) / len(node.frame_intervals)
            print(f'   Average FPS: {1.0/avg:.1f}')
            print(f'   Average interval: {avg*1000:.1f}ms')
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()