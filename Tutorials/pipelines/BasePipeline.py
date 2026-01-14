import gi
import queue as Q
from gi.repository import Gst, GstApp, GLib
import numpy as np
import time

class BasePipeline:
    def __init__(self, uri, image_queue):
        self.uri = uri
        self.pipeline = None
        self.bus = None
        self.loop = None
        self.image_queue = image_queue
        
        self.rate = 1
        
        # Common Elements
        self.queue = Gst.ElementFactory.make("queue", "queue")
        self.capsfilter_nv12 = Gst.ElementFactory.make("capsfilter", "capsfilter_nv12")
        self.capsfilter_nv12.set_property("caps", Gst.Caps.from_string("video/x-raw, format=NV12"))
        self.videoconvert = Gst.ElementFactory.make("qtivtransform", "qtivtransform")
        self.videoconvert.set_property("engine", "fcv")
        self.capsfilter_rgb = Gst.ElementFactory.make("capsfilter", "capsfilter_rgb")
        self.capsfilter_rgb.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))
        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.connect("new-sample", self.on_new_sample)
        
        self.pipeline = Gst.Pipeline.new(self.uri)
        
        # Check if common elements are created successfully
        if not all([self.queue, self.capsfilter_nv12, self.videoconvert, self.capsfilter_rgb, self.appsink]):
            print("Not all common elements could be created")
            return
        
        print("Created all common elements successfully")
        
    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("------------EOS--------------------------")
            self.reconnect()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Warning: {warn}, {debug}")
        elif t == Gst.MessageType.BUFFERING:
            percent = message.parse_buffering()
            print(f"Buffering: {percent}%")
            if percent < 100:
                self.pipeline.set_state(Gst.State.PAUSED)
            else:
                self.pipeline.set_state(Gst.State.PLAYING)

    def reconnect(self):
        print("Reconnecting pipeline...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.READY)
            time.sleep(1)
            self.pipeline.set_state(Gst.State.PLAYING)
        
    def start(self):
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.loop = GLib.MainLoop()
            try:
                self.loop.run()
            except Exception as e:
                print(f"Main loop exited: {e}")
                self.destroy()
            
    def destroy(self):
        # Clean up
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            print("Pipeline set to NULL (stopped)")

    def on_new_sample(self, appsink, data=None):
        # Callback when a new sample (frame) is available from appsink
        sample = self.appsink.pull_sample()
        if isinstance(sample, Gst.Sample):
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            
            # Extract the width, height, and format
            width = structure.get_value("width")
            height = structure.get_value("height")
            format_str = structure.get_value("format")
            
            # RGB format has 3 channels
            channels = 3
            
            # Map the buffer properly - handle stride if present
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                try:
                    # Get buffer size and check for stride
                    buffer_size = map_info.size
                    expected_size = height * width * channels
                    
                    # Check if buffer has stride (padding)
                    if buffer_size > expected_size:
                        # Buffer has stride - need to copy row by row
                        stride = buffer_size // height
                        np_array = np.zeros((height, width, channels), dtype=np.uint8)
                        
                        # Copy data row by row, skipping stride padding
                        for y in range(height):
                            row_start = y * stride
                            row_end = row_start + (width * channels)
                            row_data = bytes(map_info.data[row_start:row_end])
                            np_array[y] = np.frombuffer(row_data, dtype=np.uint8).reshape(width, channels)
                    else:
                        # No stride - direct copy
                        np_array = np.ndarray(shape=(height, width, channels),
                                              dtype=np.uint8,
                                              buffer=map_info.data[:expected_size])
                        # Make a contiguous copy
                        np_array = np.ascontiguousarray(np_array.copy())
                    
                    # Handle queue overflow by dropping the oldest frame
                    if self.image_queue.full():
                        drop_frame = self.image_queue.get()
                    
                    # Add the new frame to the queue
                    self.image_queue.put(np_array)
                    
                    return Gst.FlowReturn.OK
                finally:
                    buffer.unmap(map_info)
            else:
                print("Failed to map buffer")
                return Gst.FlowReturn.ERROR
        else:
            print("Failed to get sample")
            return Gst.FlowReturn.ERROR



