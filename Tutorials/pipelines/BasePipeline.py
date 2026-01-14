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
        # Pull the sample from appsink
        sample = self.appsink.pull_sample()
        if not isinstance(sample, Gst.Sample):
            print("Failed to get sample")
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)

        # Extract frame info
        width = structure.get_value("width")
        height = structure.get_value("height")
        format_str = structure.get_value("format")

        # Assuming RGB
        channels = 3

        # Map buffer for reading
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            print("Failed to map buffer")
            return Gst.FlowReturn.ERROR

        try:
            buffer_size = map_info.size
            expected_size = height * width * channels

            # Determine stride (bytes per row)
            if buffer_size > expected_size:
                # Stride present (padding)
                stride = buffer_size // height
            else:
                # No stride
                stride = width * channels

            # Create NumPy view using strides (NO Python loops)
            np_array = np.ndarray(
                shape=(height, width, channels),
                dtype=np.uint8,
                buffer=map_info.data,
                strides=(stride, channels, 1)
            )

            # Make a contiguous copy so the data is safe after unmap
            np_array = np_array.copy()

            # Drop oldest frame if queue is full (low-latency behavior)
            try:
                self.image_queue.put_nowait(np_array)
            except queue.Full:
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
                self.image_queue.put_nowait(np_array)

            return Gst.FlowReturn.OK

        finally:
            buffer.unmap(map_info)




