"""
Static Image Plugin for LEDMatrix

Display static images with automatic scaling, aspect ratio preservation,
and transparency support. Perfect for displaying logos, artwork, or custom graphics.

Features:
- Automatic image scaling to fit display
- Aspect ratio preservation
- Transparency support (PNG, RGBA)
- Configurable background color
- Multiple image format support (PNG, JPG, BMP, GIF)

API Version: 1.0.0
"""

import logging
import os
import time
import uuid
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
from pathlib import Path

from src.plugin_system.base_plugin import BasePlugin

logger = logging.getLogger(__name__)


class StaticImagePlugin(BasePlugin):
    """
    Static image display plugin for LED matrix.
    
    Supports image scaling, transparency handling, and configurable display options.
    
    Configuration options:
        image_path (str): Path to image file (relative or absolute)
        fit_to_display (bool): Auto-fit to display dimensions
        preserve_aspect_ratio (bool): Preserve aspect ratio when scaling
        background_color (list): RGB background color for transparent areas
        display_duration (float): Display duration in seconds
    """
    
    def __init__(self, plugin_id: str, config: Dict[str, Any],
                 display_manager, cache_manager, plugin_manager):
        """Initialize the static image plugin."""
        super().__init__(plugin_id, config, display_manager, cache_manager, plugin_manager)
        
        # Configuration
        self.fit_to_display = config.get('fit_to_display', True)
        self.preserve_aspect_ratio = config.get('preserve_aspect_ratio', True)
        # Handle background_color - can be list or tuple from JSON
        bg_color = config.get('background_color', [0, 0, 0])
        if isinstance(bg_color, (list, tuple)):
            self.background_color = tuple(bg_color)
        else:
            self.logger.warning(f"Invalid background_color type: {type(bg_color)}, using default")
            self.background_color = (0, 0, 0)
        
        # Enhanced image configuration
        raw_image_config = config.get('image_config', {}) or {}
        
        # Legacy support - migrate image_path to image_config if needed
        legacy_image_path = config.get('image_path')
        if legacy_image_path and not isinstance(raw_image_config, dict):
            raw_image_config = {}
        if legacy_image_path and not raw_image_config.get('images'):
            self.logger.info(f"Migrating legacy image_path to image_config: {legacy_image_path}")
            from datetime import datetime
            raw_image_config = {
                'mode': 'single',
                'rotation_mode': 'sequential',
                'images': [{
                    'id': str(uuid.uuid4()),
                    'path': legacy_image_path,
                    'uploaded_at': datetime.utcnow().isoformat() + 'Z',
                    'display_order': 0
                }]
            }
        
        self.image_config = self._normalize_image_config(raw_image_config)
        self.rotation_mode = self.image_config.get('rotation_mode', 'sequential')
        self.rotation_settings = config.get('rotation_settings', {})
        
        # Get images list and ensure it's always a list (defensive check)
        images_raw = self.image_config.get('images', [])
        if isinstance(images_raw, str):
            # If still a string after normalization, try to parse it
            try:
                import json as json_module
                parsed = json_module.loads(images_raw)
                if isinstance(parsed, list):
                    self.logger.warning("Images was still a JSON string after normalization; parsing manually")
                    self.images_list = parsed
                else:
                    self.logger.error(f"Parsed images string but got non-list: {type(parsed)}")
                    self.images_list = []
            except (json_module.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse images string: {e}")
                self.images_list = []
        elif isinstance(images_raw, list):
            self.images_list = images_raw
        else:
            self.logger.warning(f"Images is unexpected type {type(images_raw)}, defaulting to empty list")
            self.images_list = []
        
        # Rotation state
        self.current_image_index = 0
        self.last_rotation_time = time.time()
        
        # Setup rotation
        self._setup_rotation()
        
        # State
        self.current_image = None
        self.image_loaded = False
        self.image_path = None  # Will be set by _get_next_image()
        self.last_update_time = 0
        
        # Load initial image
        self._load_current_image()
        
        self.logger.info(f"Static image plugin initialized with {len(self.images_list)} image(s), rotation: {self.rotation_mode}")

        # Register fonts
        self._register_fonts()

    def _normalize_image_config(self, image_config: Any) -> Dict[str, Any]:
        """Normalize image configuration structure for backward compatibility."""
        normalized: Dict[str, Any]
        if isinstance(image_config, dict):
            normalized = dict(image_config)
        elif isinstance(image_config, list):
            self.logger.warning("Image config provided as list; wrapping in default config")
            normalized = {'mode': 'multiple', 'rotation_mode': 'sequential', 'images': image_config}
        elif isinstance(image_config, str):
            # Try to parse as JSON first (in case it was double-encoded)
            try:
                import json as json_module
                parsed = json_module.loads(image_config)
                if isinstance(parsed, dict):
                    self.logger.warning("Image config provided as JSON string; parsing to dict")
                    normalized = parsed
                elif isinstance(parsed, list):
                    self.logger.warning("Image config provided as JSON array string; wrapping in config")
                    normalized = {'mode': 'multiple', 'rotation_mode': 'sequential', 'images': parsed}
                else:
                    # Not a valid JSON structure, treat as image path
                    self.logger.warning("Image config provided as string; treating as single image path")
                    normalized = {
                        'mode': 'single',
                        'rotation_mode': 'sequential',
                        'images': [image_config]
                    }
            except (json_module.JSONDecodeError, ValueError):
                # Not valid JSON, treat as image path
                self.logger.warning("Image config provided as string; treating as single image path")
                normalized = {
                    'mode': 'single',
                    'rotation_mode': 'sequential',
                    'images': [image_config]
                }
        elif image_config is None:
            normalized = {}
        else:
            self.logger.warning(
                f"Unexpected image_config type ({type(image_config).__name__}); defaulting to empty config"
            )
            normalized = {}

        images_raw = normalized.get('images')
        normalized_images: List[Dict[str, Any]] = []

        if images_raw is None:
            normalized_images = []
        elif isinstance(images_raw, list):
            for index, entry in enumerate(images_raw):
                if isinstance(entry, dict):
                    normalized_images.append(entry)
                elif isinstance(entry, str):
                    # Try to parse as JSON first (in case it was double-encoded)
                    try:
                        import json as json_module
                        parsed = json_module.loads(entry)
                        if isinstance(parsed, dict):
                            self.logger.warning("Image entry provided as JSON string; parsing to dict")
                            normalized_images.append(parsed)
                        elif isinstance(parsed, list):
                            # List of image objects encoded as string
                            self.logger.warning("Image entries provided as JSON array string; unpacking")
                            for parsed_entry in parsed:
                                if isinstance(parsed_entry, dict):
                                    normalized_images.append(parsed_entry)
                        else:
                            # Not a structured object, treat as path
                            self.logger.warning("Image config entry provided as string; converting to structured record")
                            normalized_images.append({
                                'id': str(uuid.uuid4()),
                                'path': entry,
                                'display_order': index
                            })
                    except (json_module.JSONDecodeError, ValueError):
                        # Not valid JSON, treat as path
                        self.logger.warning("Image config entry provided as string; converting to structured record")
                        normalized_images.append({
                            'id': str(uuid.uuid4()),
                            'path': entry,
                            'display_order': index
                        })
                else:
                    self.logger.warning(
                        f"Unsupported image entry type ({type(entry).__name__}); skipping"
                    )
        elif isinstance(images_raw, dict):
            self.logger.warning("Image config 'images' provided as object; wrapping in list")
            normalized_images.append(images_raw)
        elif isinstance(images_raw, str):
            # Try to parse as JSON array first
            try:
                import json as json_module
                parsed = json_module.loads(images_raw)
                if isinstance(parsed, list):
                    self.logger.warning("Image config 'images' provided as JSON array string; parsing")
                    for entry in parsed:
                        if isinstance(entry, dict):
                            normalized_images.append(entry)
                        else:
                            self.logger.warning(f"Skipping non-dict entry in parsed images array: {type(entry)}")
                elif isinstance(parsed, dict):
                    self.logger.warning("Image config 'images' provided as JSON object string; parsing and wrapping")
                    normalized_images.append(parsed)
                else:
                    # Not a valid structure, treat as path
                    self.logger.warning("Image config 'images' provided as string; converting to list")
                    normalized_images.append({
                        'id': str(uuid.uuid4()),
                        'path': images_raw,
                        'display_order': 0
                    })
            except (json_module.JSONDecodeError, ValueError):
                # Not valid JSON, treat as path
                self.logger.warning("Image config 'images' provided as string; converting to list")
                normalized_images.append({
                    'id': str(uuid.uuid4()),
                    'path': images_raw,
                    'display_order': 0
                })
        else:
            self.logger.warning(
                f"Unsupported images field type ({type(images_raw).__name__}); defaulting to empty list"
            )

        normalized['images'] = normalized_images
        return normalized

    def _register_fonts(self):
        """Register fonts with the font manager."""
        try:
            if not hasattr(self.plugin_manager, 'font_manager'):
                return

            font_manager = self.plugin_manager.font_manager

            # Error message font
            font_manager.register_manager_font(
                manager_id=self.plugin_id,
                element_key=f"{self.plugin_id}.error",
                family="press_start",
                size_px=8,
                color=(255, 0, 0)  # Red for errors
            )

            # Info font
            font_manager.register_manager_font(
                manager_id=self.plugin_id,
                element_key=f"{self.plugin_id}.info",
                family="four_by_six",
                size_px=6,
                color=(150, 150, 150)
            )

            self.logger.info("Static image fonts registered")
        except Exception as e:
            self.logger.warning(f"Error registering fonts: {e}")

    def _resolve_image_path(self, image_path: str) -> str:
        """
        Resolve image path to absolute path.
        Handles both absolute paths and relative paths (from project root).
        
        Args:
            image_path: Image path from config (may be relative or absolute)
            
        Returns:
            Absolute path to image file
        """
        if not image_path:
            return None
            
        # If already absolute, check if it exists
        if os.path.isabs(image_path):
            if os.path.exists(image_path):
                return image_path
            # If absolute path doesn't exist, try relative to project root
            # (path might have been saved as absolute but project moved)
        
        # Try relative to current working directory
        if os.path.exists(image_path):
            return os.path.abspath(image_path)
        
        # Try relative to project root (common for uploaded assets)
        # Project root is typically the directory containing run.py
        project_root = Path(__file__).resolve().parent.parent.parent
        project_path = project_root / image_path
        if project_path.exists():
            return str(project_path)
        
        # Try as-is in case it's already resolved
        return image_path
    
    def _load_image(self) -> bool:
        """
        Load and process the image for display.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.image_path:
            self.logger.warning("No image path specified")
            return False
            
        # Resolve path (handles relative paths from project root)
        resolved_path = self._resolve_image_path(self.image_path)
        
        if not resolved_path or not os.path.exists(resolved_path):
            self.logger.warning(f"Image file not found: {self.image_path} (resolved: {resolved_path})")
            # Try to list what's in the uploads directory for debugging
            project_root = Path(__file__).resolve().parent.parent.parent
            uploads_dir = project_root / "assets" / "plugins" / self.plugin_id / "uploads"
            if uploads_dir.exists():
                files = list(uploads_dir.glob("*.*"))
                self.logger.info(f"Available files in uploads directory: {[f.name for f in files]}")
            return False
        
        # Use resolved path for loading
        actual_image_path = resolved_path
        
        try:
            # Load the image using resolved path
            img = Image.open(actual_image_path)
            
            # Convert to RGBA to handle transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Get display dimensions
            display_width = self.display_manager.matrix.width
            display_height = self.display_manager.matrix.height
            
            # Calculate target size
            if self.fit_to_display and self.preserve_aspect_ratio:
                target_size = self._calculate_fit_size(img.size, (display_width, display_height))
            elif self.fit_to_display:
                target_size = (display_width, display_height)
            else:
                target_size = img.size
            
            # Resize image if needed
            if target_size != img.size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Create display-sized canvas with background color
            canvas = Image.new('RGB', (display_width, display_height), self.background_color)
            
            # Calculate position to center the image
            paste_x = (display_width - img.width) // 2
            paste_y = (display_height - img.height) // 2
            
            # Handle transparency by compositing
            if img.mode == 'RGBA':
                # Create a temporary canvas with background color
                temp_canvas = Image.new('RGB', (display_width, display_height), self.background_color)
                temp_canvas.paste(img, (paste_x, paste_y), img)
                canvas = temp_canvas
            else:
                canvas.paste(img, (paste_x, paste_y))
            
            self.current_image = canvas
            self.image_loaded = True
            self.last_update_time = time.time()
            
            self.logger.info(f"Successfully loaded and processed image: {self.image_path} -> {actual_image_path}")
            # Reopen for size info (using resolved path)
            size_img = Image.open(actual_image_path)
            self.logger.info(f"Original size: {size_img.size}, "
                           f"Display size: {target_size}, Position: ({paste_x}, {paste_y})")
            size_img.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading image {self.image_path}: {e}")
            self.image_loaded = False
            return False
    
    def _setup_rotation(self) -> None:
        """Initialize rotation based on mode"""
        if self.rotation_mode == 'random':
            import random
            seed = self.rotation_settings.get('random_seed')
            if seed is not None:
                random.seed(seed)
        
        # Sort images by display_order if available
        if self.images_list:
            self.images_list.sort(key=lambda x: x.get('display_order', 0))
    
    def _is_image_scheduled(self, image_info: Dict[str, Any]) -> bool:
        """
        Check if an image should be displayed based on its schedule.
        
        Args:
            image_info: Image dictionary with optional schedule field
            
        Returns:
            True if image should be displayed now, False otherwise
        """
        schedule = image_info.get('schedule')
        if not schedule or not schedule.get('enabled', False):
            # No schedule or schedule disabled = always show
            return True
        
        mode = schedule.get('mode', 'always')
        if mode == 'always':
            return True
        
        # Get current time
        from datetime import datetime
        now = datetime.now()
        current_time = now.time()
        current_day = now.strftime('%A').lower()  # monday, tuesday, etc.
        
        if mode == 'time_range':
            # Same time every day
            start_time_str = schedule.get('start_time', '08:00')
            end_time_str = schedule.get('end_time', '18:00')
            
            try:
                start_time = datetime.strptime(start_time_str, '%H:%M').time()
                end_time = datetime.strptime(end_time_str, '%H:%M').time()
                
                # Handle times that span midnight
                if start_time <= end_time:
                    return start_time <= current_time <= end_time
                else:
                    # Spans midnight (e.g., 22:00 - 06:00)
                    return current_time >= start_time or current_time <= end_time
            except Exception as e:
                self.logger.warning(f"Error parsing schedule times: {e}")
                return True
        
        elif mode == 'per_day':
            # Different times per day
            days = schedule.get('days', {})
            day_config = days.get(current_day)
            
            if not day_config or not day_config.get('enabled', True):
                return False
            
            start_time_str = day_config.get('start_time', '08:00')
            end_time_str = day_config.get('end_time', '18:00')
            
            try:
                start_time = datetime.strptime(start_time_str, '%H:%M').time()
                end_time = datetime.strptime(end_time_str, '%H:%M').time()
                
                # Handle times that span midnight
                if start_time <= end_time:
                    return start_time <= current_time <= end_time
                else:
                    # Spans midnight
                    return current_time >= start_time or current_time <= end_time
            except Exception as e:
                self.logger.warning(f"Error parsing day schedule times: {e}")
                return True
        
        # Unknown mode - default to showing
        return True
    
    def _get_available_images(self) -> List[Dict[str, Any]]:
        """Get list of images that are currently scheduled to be shown."""
        available = []
        for img in self.images_list:
            if self._is_image_scheduled(img):
                available.append(img)
        return available if available else self.images_list  # Fallback to all if none scheduled
    
    def _get_next_image(self) -> Optional[Dict[str, Any]]:
        """Get next image based on rotation mode and schedule"""
        if not self.images_list:
            return None
        
        # Get available images (filtered by schedule)
        available_images = self._get_available_images()
        
        if not available_images:
            # No images available right now - return first unscheduled image as fallback
            self.logger.debug("No scheduled images available, using first image as fallback")
            return self.images_list[0] if self.images_list else None
        
        if self.rotation_mode == 'sequential':
            # Find which images in the full list are available
            available_indices = [i for i, img in enumerate(self.images_list) if img in available_images]
            if not available_indices:
                return available_images[0]
            
            # Find the current image index in available images
            current_available_idx = 0
            for i, idx in enumerate(available_indices):
                if idx >= self.current_image_index:
                    current_available_idx = i
                    break
            
            image = available_images[current_available_idx]
            sequential_loop = self.rotation_settings.get('sequential_loop', True)
            
            # Move to next available image
            next_available_idx = (current_available_idx + 1) % len(available_images) if sequential_loop else min(current_available_idx + 1, len(available_images) - 1)
            
            # Update current_image_index to match the next image in full list
            if next_available_idx < len(available_indices):
                next_full_idx = available_indices[next_available_idx]
                self.current_image_index = next_full_idx
            elif sequential_loop and next_available_idx == 0:
                # Looped back to start
                self.current_image_index = available_indices[0]
            
            return image
        
        elif self.rotation_mode == 'random':
            import random
            return random.choice(available_images)
        
        elif self.rotation_mode == 'time_based':
            time_intervals = self.rotation_settings.get('time_intervals', {})
            if time_intervals.get('enabled', False):
                interval_seconds = time_intervals.get('interval_seconds', 3600)
                now = time.time()
                if now - self.last_rotation_time >= interval_seconds:
                    # Move to next available image
                    available_indices = [i for i, img in enumerate(self.images_list) if img in available_images]
                    if available_indices:
                        current_idx_in_available = available_indices.index(self.current_image_index) if self.current_image_index in available_indices else 0
                        next_idx_in_available = (current_idx_in_available + 1) % len(available_images)
                        if next_idx_in_available < len(available_images):
                            next_image = available_images[next_idx_in_available]
                            self.current_image_index = next((i for i, img in enumerate(self.images_list) if img == next_image), 0)
                    self.last_rotation_time = now
            
            # Return current image from available
            available_indices = [i for i, img in enumerate(self.images_list) if img in available_images]
            if self.current_image_index in available_indices:
                return self.images_list[self.current_image_index]
            else:
                return available_images[0]
        
        elif self.rotation_mode == 'date_based':
            # Future implementation
            return available_images[0]
        
        # Default: return first available image
        return available_images[0]
    
    def _load_current_image(self) -> bool:
        """Load the current image based on rotation"""
        image_info = self._get_next_image()
        if not image_info:
            self.logger.warning("No images available for display")
            return False
        
        image_path = image_info.get('path')
        if not image_path:
            self.logger.warning("Image entry has no path")
            return False
        
        self.image_path = image_path
        return self._load_image()
    
    def _calculate_fit_size(self, image_size: Tuple[int, int], 
                           display_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate size to fit image within display bounds while preserving aspect ratio.
        
        Args:
            image_size: Original image dimensions (width, height)
            display_size: Display dimensions (width, height)
        
        Returns:
            Calculated dimensions (width, height)
        """
        img_width, img_height = image_size
        display_width, display_height = display_size
        
        # Calculate scaling factor to fit within display
        scale_x = display_width / img_width
        scale_y = display_height / img_height
        scale = min(scale_x, scale_y)
        
        return (int(img_width * scale), int(img_height * scale))
    
    def update(self) -> None:
        """
        Update method - no continuous updates needed for static images.
        
        Static images don't change, so this method is a no-op.
        """
        pass
    
    def display(self, force_clear: bool = False) -> None:
        """
        Display the static image on the LED matrix.
        
        Args:
            force_clear: If True, clear display before rendering
        """
        # Check if we need to reload image based on schedule changes
        # Schedules are checked in _get_next_image, but we need to reload
        # when schedules might have changed (time-based or random modes)
        should_reload = False
        
        if self.rotation_mode in ['random']:
            # Random mode - check schedule on each call
            should_reload = True
        elif self.rotation_mode == 'time_based':
            # Check if we should rotate based on time intervals
            time_intervals = self.rotation_settings.get('time_intervals', {})
            if time_intervals.get('enabled', False):
                should_reload = True
        elif self.rotation_mode == 'sequential':
            # For sequential, only reload if current image is no longer scheduled
            if self.image_path and self.images_list:
                current_image_info = next((img for img in self.images_list if img.get('path') == self.image_path), None)
                if current_image_info:
                    if not self._is_image_scheduled(current_image_info):
                        # Current image is no longer scheduled, reload
                        should_reload = True
        
        # Reload image if needed
        if should_reload:
            if not self._load_current_image():
                # Try to find any available scheduled image
                available = self._get_available_images()
                if available and available != self.images_list:
                    # Found scheduled images, load one
                    self._load_current_image()
                else:
                    self._display_error()
                    return
        
        if not self.image_loaded or not self.current_image:
            self.logger.warning("No image loaded for display")
            self._display_error()
            return
        
        try:
            # Clear display if requested
            if force_clear:
                self.display_manager.clear()
            
            # Set the image on the display manager
            self.display_manager.image = self.current_image.copy()
            
            # Update the display
            self.display_manager.update_display()
            
            self.logger.debug(f"Displayed image: {self.image_path} (mode: {self.rotation_mode})")
            
        except Exception as e:
            self.logger.error(f"Error displaying image: {e}")
            self._display_error()
    
    def _display_error(self) -> None:
        """Display error message when image can't be loaded."""
        try:
            # Get error font from font manager
            error_font = None
            try:
                if hasattr(self.plugin_manager, 'font_manager'):
                    font_manager = self.plugin_manager.font_manager
                    error_font = font_manager.resolve_font(
                        element_key=f"{self.plugin_id}.error",
                        family="press_start",
                        size_px=8
                    )
            except Exception as e:
                self.logger.warning(f"Error getting font from font manager: {e}")

            img = Image.new('RGB',
                          (self.display_manager.matrix.width,
                           self.display_manager.matrix.height),
                          (0, 0, 0))

            if error_font:
                self.display_manager.image = img.copy()
                self.display_manager.draw_text("Image", x=5, y=12, font=error_font, centered=False)
                self.display_manager.draw_text("Error", x=5, y=20, font=error_font, centered=False)
            else:
                # Fallback to direct PIL if font manager fails
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)

                try:
                    font = ImageFont.truetype('assets/fonts/4x6-font.ttf', 8)
                except:
                    font = ImageFont.load_default()

                draw.text((5, 12), "Image", font=font, fill=(200, 0, 0))
                draw.text((5, 20), "Error", font=font, fill=(200, 0, 0))

                self.display_manager.image = img.copy()

            self.display_manager.update_display()
        except Exception as e:
            self.logger.error(f"Error displaying error message: {e}")
            pass
    
    def set_image_path(self, image_path: str) -> bool:
        """
        Set a new image path and load it.
        
        Args:
            image_path: Path to new image file
        
        Returns:
            True if successful, False otherwise
        """
        self.image_path = image_path
        return self._load_image()
    
    def reload_image(self) -> bool:
        """
        Reload the current image.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.image_path:
            self.logger.warning("No image path set for reload")
            return False
        
        return self._load_image()
    
    def get_display_duration(self) -> float:
        """Get display duration from config."""
        return self.config.get('display_duration', 10.0)
    
    def validate_config(self) -> bool:
        """Validate plugin configuration."""
        # Call parent validation first
        if not super().validate_config():
            return False
        
        # Validate images list
        if self.images_list:
            for img in self.images_list:
                img_path = img.get('path')
                if not img_path:
                    self.logger.warning("Image entry missing path")
                    continue
                if not os.path.exists(img_path):
                    self.logger.warning(f"Image file not found: {img_path}")
                    # Don't fail validation, just warn - file might be added later
        elif not self.image_path:
            self.logger.warning("No images configured and no legacy image_path specified")
            # Don't fail - might be uploading images
        
        # Validate background color (can be list or tuple from config)
        bg_color = self.config.get('background_color', [0, 0, 0])
        if not isinstance(bg_color, (list, tuple)) or len(bg_color) != 3:
            self.logger.error("Invalid background_color: must be RGB list or tuple with 3 values")
            return False
        
        if not all(isinstance(c, (int, float)) and 0 <= c <= 255 for c in bg_color):
            self.logger.error("Invalid background_color: values must be numbers between 0-255")
            return False
        
        return True
    
    def on_config_change(self, new_config: Dict[str, Any]) -> None:
        """Called after plugin configuration has been updated via the web API."""
        super().on_config_change(new_config)
        
        # Update image configuration
        old_images_count = len(self.images_list)
        old_rotation_mode = self.rotation_mode
        
        raw_image_config = self.config.get('image_config', {}) or {}
        self.image_config = self._normalize_image_config(raw_image_config)
        self.rotation_mode = self.image_config.get('rotation_mode', 'sequential')
        self.rotation_settings = self.config.get('rotation_settings', {})
        self.images_list = self.image_config.get('images', [])
        
        # Reinitialize rotation
        self._setup_rotation()
        
        # Reload image if configuration changed
        if len(self.images_list) != old_images_count or self.rotation_mode != old_rotation_mode:
            self.current_image_index = 0  # Reset index
            self._load_current_image()
            self.logger.info(f"Config updated: {len(self.images_list)} images, rotation: {self.rotation_mode}")
    
    def get_info(self) -> Dict[str, Any]:
        """Return plugin info for web UI."""
        info = super().get_info()
        info.update({
            'image_path': self.image_path,
            'image_loaded': self.image_loaded,
            'fit_to_display': self.fit_to_display,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'background_color': self.background_color,
            'rotation_mode': self.rotation_mode,
            'images_count': len(self.images_list)
        })
        
        if self.current_image:
            info['display_size'] = self.current_image.size
        
        return info
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.current_image = None
        self.image_loaded = False
        self.logger.info("Static image plugin cleaned up")

