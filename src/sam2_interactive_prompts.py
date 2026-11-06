"""
sam2_interactive_prompts.py - Click to select holes for SAM2
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict
from pathlib import Path

class InteractiveHoleSelector:
    """
    Interactive UI to click on holes in the first frame
    """

    def __init__(self):
        self.points = []
        self.fig = None
        self.ax = None
        self.scatter = None
        # Timestamp of the first left click this selection round -- None
        # until clicked. Lets the caller time from "user actually engaged"
        # rather than "window opened", since the window can sit open for a
        # while before anyone's available to click it.
        self.first_click_time = None
        
    def select_holes(self, 
                     bg_render: torch.Tensor,
                     max_holes: int = 10) -> List[Dict]:
        """
        Show interactive UI to click on holes
        
        Controls:
        - LEFT CLICK: Add hole point
        - RIGHT CLICK: Remove last point
        - ENTER/CLOSE: Finish selection
        
        Args:
            bg_render: (3, H, W) First background render
            max_holes: Maximum number of holes to select
        
        Returns:
            hole_prompts: List of {point, label, object_id}
        """
        self.first_click_time = None

        # Convert to displayable image
        img_np = bg_render.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.clip(img_np, 0, 1)
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(img_np)
        self.ax.set_title(
            'Click on HOLES (dark areas that need inpainting)\n'
            'LEFT CLICK = Add hole | RIGHT CLICK = Remove last | ENTER = Finish',
            fontsize=12, pad=15
        )
        self.ax.axis('off')
        
        # Initialize scatter plot for points
        self.scatter = self.ax.scatter([], [], c='red', s=200, marker='x', linewidths=3)
        
        # Text annotations
        self.annotations = []
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n" + "="*70)
        print("INTERACTIVE HOLE SELECTION")
        print("="*70)
        print("Instructions:")
        print("  - LEFT CLICK on each hole (dark area that needs inpainting)")
        print("  - RIGHT CLICK to remove the last point")
        print("  - PRESS ENTER or CLOSE WINDOW when done")
        print(f"  - Select up to {max_holes} holes")
        print("="*70 + "\n")
        
        plt.tight_layout()
        plt.show()
        
        # Convert points to prompts
        hole_prompts = []
        for i, (x, y) in enumerate(self.points):
            hole_prompts.append({
                'point': np.array([x, y], dtype=np.float32),
                'label': 1,  # Foreground
                'object_id': i + 1  # Start from 1
            })
        
        print(f"\n✓ Selected {len(hole_prompts)} holes")
        
        return hole_prompts
    
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click - add point
            if self.first_click_time is None:
                self.first_click_time = time.time()
            x, y = event.xdata, event.ydata
            self.points.append((x, y))
            
            print(f"  Hole {len(self.points)}: ({x:.0f}, {y:.0f})")
            
            # Update visualization
            self.update_points()
            
        elif event.button == 3:  # Right click - remove last
            if self.points:
                removed = self.points.pop()
                print(f"  Removed: ({removed[0]:.0f}, {removed[1]:.0f})")
                self.update_points()
    
    def on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'enter':
            plt.close(self.fig)
        elif event.key == 'backspace':
            if self.points:
                removed = self.points.pop()
                print(f"  Removed: ({removed[0]:.0f}, {removed[1]:.0f})")
                self.update_points()
    
    def update_points(self):
        """Update the visualization of selected points"""
        if not self.points:
            self.scatter.set_offsets(np.empty((0, 2)))
            for ann in self.annotations:
                ann.remove()
            self.annotations = []
        else:
            # Update scatter plot
            points_array = np.array(self.points)
            self.scatter.set_offsets(points_array)
            
            # Update annotations
            for ann in self.annotations:
                ann.remove()
            self.annotations = []
            
            for i, (x, y) in enumerate(self.points):
                ann = self.ax.annotate(
                    f'{i+1}',
                    (x, y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=14,
                    fontweight='bold',
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )
                self.annotations.append(ann)
        
        self.fig.canvas.draw()