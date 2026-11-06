"""
sam2_interactive_prompts_cv.py - OpenCV-based interactive selector
"""

import time
import cv2
import numpy as np
import torch
from typing import List, Dict

class InteractiveHoleSelectorCV:
    """
    OpenCV-based interactive hole selector (more responsive)
    """

    def __init__(self):
        self.points = []
        self.display_img = None
        # Timestamp of the first left click this selection round -- None
        # until clicked. Lets the caller time from "user actually engaged"
        # rather than "window opened", since the window can sit open for a
        # while before anyone's available to click it.
        self.first_click_time = None
        
    def select_holes(self, 
                     bg_render: torch.Tensor,
                     window_name: str = "Select Holes") -> List[Dict]:
        """
        Show interactive OpenCV window to click on holes
        
        Controls:
        - LEFT CLICK: Add hole point
        - RIGHT CLICK: Remove last point
        - SPACE/ENTER: Finish selection
        - ESC: Cancel
        
        Args:
            bg_render: (3, H, W) First background render
            window_name: Window title
        
        Returns:
            hole_prompts: List of {point, label, object_id}
        """
        self.first_click_time = None

        # Convert to OpenCV format
        img_np = bg_render.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        self.display_img = img_np.copy()
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        
        # Set mouse callback
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*70)
        print("INTERACTIVE HOLE SELECTION (OpenCV)")
        print("="*70)
        print("Instructions:")
        print("  - LEFT CLICK on each hole (dark area that needs inpainting)")
        print("  - RIGHT CLICK to remove the last point")
        print("  - SPACE or ENTER to finish")
        print("  - ESC to cancel")
        print("="*70 + "\n")
        
        while True:
            # Draw current state
            display = self.display_img.copy()
            
            # Draw all points
            for i, (x, y) in enumerate(self.points):
                # Draw cross marker
                cv2.drawMarker(
                    display, (int(x), int(y)),
                    color=(0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=30,
                    thickness=3
                )
                
                # Draw number label
                cv2.putText(
                    display, f'{i+1}',
                    (int(x) + 15, int(y) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2
                )
            
            # Draw instructions
            instructions = [
                f"Holes selected: {len(self.points)}",
                "LEFT CLICK = Add | RIGHT CLICK = Remove | SPACE = Finish | ESC = Cancel"
            ]
            
            y_offset = 30
            for text in instructions:
                cv2.putText(
                    display, text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                y_offset += 35
            
            cv2.imshow(window_name, display)
            
            # Wait for key
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - cancel
                print("Cancelled")
                self.points = []
                break
            elif key == 13 or key == 32:  # ENTER or SPACE - finish
                break
        
        cv2.destroyAllWindows()
        
        # Convert points to prompts
        hole_prompts = []
        for i, (x, y) in enumerate(self.points):
            hole_prompts.append({
                'point': np.array([x, y], dtype=np.float32),
                'label': 1,
                'object_id': i + 1
            })
        
        print(f"\n✓ Selected {len(hole_prompts)} holes")
        for i, prompt in enumerate(hole_prompts):
            print(f"  Hole {i+1}: ({prompt['point'][0]:.0f}, {prompt['point'][1]:.0f})")
        
        return hole_prompts
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click - add point
            if self.first_click_time is None:
                self.first_click_time = time.time()
            self.points.append((x, y))
            print(f"  Hole {len(self.points)}: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click - remove last
            if self.points:
                removed = self.points.pop()
                print(f"  Removed: ({removed[0]}, {removed[1]})")