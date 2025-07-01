import os
import time
import threading
import queue
import struct
import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fftpack
from scipy.signal import decimate
import sounddevice as sd
import serial
import psutil
import logging

# Settings
RECORD_SECONDS = 5
SAMPLE_RATE = 44100
AUDIO_DIR = "./recordings"
UART_PORT = "/dev/serial0"  # Change to your ESP32 port (e.g., "/dev/ttyUSB0")
UART_BAUDRATE = 230400
CLEAN_INTERVAL_DAYS = 7
DECIMATED_RATE = 500
EVENTS_PER_CYCLE = 2
DISK_SPACE_THRESHOLD = 0.1
MIN_FREE_SPACE_GB = 1.0
ANALYZE_QUEUE_SIZE = 10
MIC_DEVICE = None
LOG_FILE = "audio_system.log"

# UART Protocol Constants
START_BYTE_DATA = 0xAA
START_BYTE_RECORD = 0xAC
START_BYTE_SYNC = 0xAB
END_BYTE = 0x55
ACK_TIMEOUT = 1.0
MAX_RETRIES = 5
INTER_PACKET_DELAY = 0.05

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs(AUDIO_DIR, exist_ok=True)

# Queues
analyze_queue = queue.Queue(maxsize=ANALYZE_QUEUE_SIZE)

# Serial UART init with better error handling
ser = None
uart_lock = threading.Lock()

def init_uart():
    global ser
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            ser = serial.Serial(
                UART_PORT, 
                UART_BAUDRATE, 
                timeout=1, 
                rtscts=False, 
                xonxoff=False, 
                write_timeout=2.0
            )
            time.sleep(2)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            print(f"✓ UART connected on {UART_PORT} at {UART_BAUDRATE} baud")
            logging.info(f"UART connected on {UART_PORT} at {UART_BAUDRATE} baud")
            return True
        except Exception as e:
            print(f"✗ UART connection failed (attempt {attempt + 1}/{max_attempts}): {e}")
            logging.error(f"UART connection failed: {e}")
            ser = None
            time.sleep(5)
    print("✗ Fatal: Could not initialize UART. Halting system.")
    logging.critical("Could not initialize UART. Halting system.")
    exit(1)

file_counter = 1
last_cleanup = time.time()
packet_seq_num = 0

# Datum classes (unchanged)
class RawDatumKey:
    AUDIO_ARRAY = "audio_array"
    SAMPLE_RATE = "sample_rate"

class DerivedDataKey:
    DECIMATED_AUDIO = "decimated_audio"
    DECIMATED_AUDIO_RATE = "decimated_audio_rate"
    FREQUENCY_PEAK = "frequency_peak"
    DECIMATED_FREQUENCY_PEAK = "decimated_frequency_peak"
    RPM = "rpm"
    ENGINE_STATUS = "engine_status"

class Datum:
    def __init__(self, audio_array, sample_rate):
        self._raw_data = {RawDatumKey.AUDIO_ARRAY: audio_array, RawDatumKey.SAMPLE_RATE: sample_rate}
        self._derived_data = {}

    def get_raw_datum(self, key):
        return self._raw_data.get(key)

    def set_raw_datum(self, key, value):
        self._raw_data[key] = value

    def get_derived_data(self, key):
        return self._derived_data.get(key)

    def set_derived_data(self, key, value):
        self._derived_data[key] = value

# FeatureExtraction interface (unchanged)
class FeatureExtraction:
    def extract_features(self, datum: Datum) -> Datum:
        raise NotImplementedError

# Feature engineering blocks (unchanged)
class Decimation(FeatureExtraction):
    def __init__(self, target_rate=500):
        self.target_rate = target_rate

    def extract_features(self, datum: Datum):
        signal = datum.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
        original_rate = datum.get_raw_datum(RawDatumKey.SAMPLE_RATE)
        try:
            decimation_factor = int(original_rate // self.target_rate)
            if decimation_factor < 1:
                decimated_signal = signal
                decimated_rate = original_rate
            else:
                decimated_signal = decimate(signal, decimation_factor, axis=0, ftype='iir')
                decimated_rate = original_rate // decimation_factor
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO, decimated_signal)
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, decimated_rate)
        except Exception as e:
            print(f"✗ Decimation error: {e}")
            logging.error(f"Decimation error: {e}")
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO, signal)
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, original_rate)
        return datum

class FrequencyPeakFinder(FeatureExtraction):
    def __init__(self, buffer_seconds=5):
        self.buffer_seconds = buffer_seconds

    def extract_features(self, datum: Datum):
        try:
            signal = datum.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
            sample_rate = datum.get_raw_datum(RawDatumKey.SAMPLE_RATE)
            windowed_signal = signal * np.hanning(len(signal))
            N = len(windowed_signal)
            fft_data = np.abs(fftpack.fft(windowed_signal)[:N//2])
            freqs = fftpack.fftfreq(N, 1/sample_rate)[:N//2]
            valid_range = (freqs >= 10) & (freqs <= 200)
            if np.any(valid_range):
                valid_fft = fft_data[valid_range]
                valid_freqs = freqs[valid_range]
                peak_idx = np.argmax(valid_fft)
                peak_freq = valid_freqs[peak_idx]
            else:
                peak_freq = 0.0
            datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, peak_freq)
            decimated_signal = datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO)
            decimated_rate = datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE)
            if decimated_signal is not None and len(decimated_signal) > 0:
                windowed_decimated = decimated_signal * np.hanning(len(decimated_signal))
                N = len(windowed_decimated)
                fft_data = np.abs(fftpack.fft(windowed_decimated)[:N//2])
                freqs = fftpack.fftfreq(N, 1/decimated_rate)[:N//2]
                valid_range = (freqs >= 10) & (freqs <= 200)
                if np.any(valid_range):
                    valid_fft = fft_data[valid_range]
                    valid_freqs = freqs[valid_range]
                    peak_idx = np.argmax(valid_fft)
                    decimated_peak_freq = valid_freqs[peak_idx]
                else:
                    decimated_peak_freq = 0.0
                datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, decimated_peak_freq)
            else:
                datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, 0.0)
        except Exception as e:
            print(f"✗ Peak finding error: {e}")
            logging.error(f"Peak finding error: {e}")
            datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, 0.0)
            datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, 0.0)
        return datum

class RPM(FeatureExtraction):
    def __init__(self, events_per_crankshaft_cycle=2):
        self.events_per_cycle = events_per_crankshaft_cycle

    def extract_features(self, datum: Datum):
        try:
            peak_freq = datum.get_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK)
            if peak_freq is None:
                peak_freq = 0.0
            rpm = (peak_freq * 60) / self.events_per_cycle
            if rpm < 0 or rpm > 10000:
                rpm = 0.0
            datum.set_derived_data(DerivedDataKey.RPM, rpm)
            engine_status = 1 if rpm > 100 else 0
            datum.set_derived_data(DerivedDataKey.ENGINE_STATUS, engine_status)
        except Exception as e:
            print(f"✗ RPM calculation error: {e}")
            logging.error(f"RPM calculation error: {e}")
            datum.set_derived_data(DerivedDataKey.RPM, 0.0)
            datum.set_derived_data(DerivedDataKey.ENGINE_STATUS, 0)
        return datum

# FeatureEngineeringPipeline (unchanged)
class FeatureEngineeringPipeline:
    def __init__(self):
        self.__feature_engineering_blocks = []
        self.__output_blocks = []

    def add_block(self, block: FeatureExtraction) -> None:
        self.__feature_engineering_blocks.append(block)

    def add_output_block(self, block: FeatureExtraction) -> None:
        self.__output_blocks.append(block)

    def run(self, datum: Datum) -> Datum:
        for block in self.__feature_engineering_blocks:
            datum = block.extract_features(datum)
        for block in self.__output_blocks:
            datum = block.extract_features(datum)
        return datum

def get_timestamp():
    # Modified: Return timestamp as milliseconds (uint64_t) for precision
    return int(time.time() * 1000)

def calculate_crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc = crc << 1
            crc &= 0xFF
    return crc

def create_uart_packet(timestamp, datum: Datum, seq_num):
    try:
        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
        engine_status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
        peak_freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
        # Modified: Use uint64_t (8 bytes) for timestamp
        payload = bytearray()
        payload.extend(struct.pack('<H', seq_num))  # 2-byte sequence number
        payload.extend(struct.pack('<Q', timestamp))  # 8-byte timestamp (uint64_t ms)
        payload.extend(struct.pack('<f', float(rpm)))  # 4-byte RPM
        payload.extend(struct.pack('<B', int(engine_status)))  # 1-byte status
        payload.extend(struct.pack('<f', float(peak_freq)))  # 4-byte frequency
        length = len(payload)
        crc = calculate_crc8(payload)
        packet = bytearray([START_BYTE_DATA, length])
        packet.extend(payload)
        packet.append(crc)
        packet.append(END_BYTE)
        return packet
    except Exception as e:
        print(f"Packet creation error: {e}")
        logging.error(f"Packet creation error: {e}")
        return None

def send_uart_packet(timestamp, datum: Datum, seq_num):
    global packet_seq_num
    # Modified: Check UART connection and reconnect if needed
    if ser is None or not ser.is_open:
        print("UART disconnected. Attempting to reconnect...")
        logging.warning("UART disconnected. Attempting to reconnect.")
        init_uart()
        if ser is None:
            return False
    with uart_lock:
        for attempt in range(MAX_RETRIES):
            try:
                packet = create_uart_packet(timestamp, datum, seq_num)
                if packet is None:
                    return False
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                bytes_written = ser.write(packet)
                ser.flush()
                if bytes_written != len(packet):
                    print(f"✗ Incomplete write: {bytes_written}/{len(packet)} bytes")
                    continue
                # Modified: Expect 3-byte ACK (0x06 + 2-byte seq_num)
                ser.timeout = ACK_TIMEOUT
                response = ser.read(3)
                if len(response) == 3 and response[0] == 0x06:
                    ack_seq = struct.unpack('<H', response[1:3])[0]
                    if ack_seq == seq_num:
                        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                        status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
                        freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
                        print(f"✓ UART OK - Seq:{seq_num}, RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
                        logging.info(f"Packet sent - Seq:{seq_num}, RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
                        return True
                    else:
                        print(f"✗ ACK sequence mismatch: expected {seq_num}, got {ack_seq}")
                        logging.warning(f"ACK sequence mismatch: expected {seq_num}, got {ack_seq}")
                else:
                    print(f"✗ No/invalid ACK (attempt {attempt + 1}/{MAX_RETRIES})")
                    logging.warning(f"No/invalid ACK received on attempt {attempt + 1}")
            except Exception as e:
                print(f"✗ UART send error (attempt {attempt + 1}): {e}")
                logging.error(f"UART send error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INTER_PACKET_DELAY * (attempt + 1))
        print(f"✗ Failed to send packet after {MAX_RETRIES} attempts")
        logging.error(f"Failed to send packet after {MAX_RETRIES} attempts")
        return False

def send_record_alert(timestamp):
    if ser is None:
        return
    with uart_lock:
        try:
            # Modified: Use uint64_t (8 bytes) for timestamp
            payload = struct.pack('<Q', timestamp)
            crc = calculate_crc8(payload)
            packet = bytearray([START_BYTE_RECORD, len(payload)])
            packet.extend(payload)
            packet.append(crc)
            packet.append(END_BYTE)
            ser.write(packet)
            ser.flush()
            print(f"Record alert sent: {timestamp}")
            logging.info(f"Record alert sent: {timestamp}")
        except Exception as e:
            print(f"✗ Record alert error: {e}")
            logging.error(f"Record alert error: {e}")

def send_sync_packet():
    if ser is None:
        return
    with uart_lock:
        try:
            # Modified: Use uint64_t (8 bytes) for timestamp
            current_time = get_timestamp()
            payload = struct.pack('<Q', current_time)
            crc = calculate_crc8(payload)
            packet = bytearray([START_BYTE_SYNC, len(payload)])
            packet.extend(payload)
            packet.append(crc)
            packet.append(END_BYTE)
            ser.write(packet)
            ser.flush()
            print(f"Time sync sent: {current_time}")
            logging.info(f"Time sync sent: {current_time}")
        except Exception as e:
            print(f"✗ Sync error: {e}")
            logging.error(f"Sync error: {e}")

def check_disk_space():
    try:
        disk = psutil.disk_usage(AUDIO_DIR)
        free_space = disk.free / (1024 ** 3)
        free_percent = disk.free / disk.total
        return free_space < MIN_FREE_SPACE_GB or free_percent < DISK_SPACE_THRESHOLD
    except Exception as e:
        print(f"✗ Disk space check error: {e}")
        return False

def cleanup_old_files():
    print("🧹 Cleaning old recordings...")
    logging.info("Cleaning old recordings")
    deleted_count = 0
    try:
        files = [(f, os.path.getmtime(os.path.join(AUDIO_DIR, f))) 
                for f in os.listdir(AUDIO_DIR) 
                if os.path.isfile(os.path.join(AUDIO_DIR, f)) and f.endswith('.wav')]
        files.sort(key=lambda x: x[1])
        for file, _ in files:
            if not check_disk_space():
                break
            path = os.path.join(AUDIO_DIR, file)
            try:
                os.remove(path)
                deleted_count += 1
                print(f"Deleted: {file}")
                logging.info(f"Deleted: {file}")
            except Exception as e:
                print(f"✗ Delete error: {e}")
                logging.error(f"Delete error: {e}")
    except Exception as e:
        print(f"✗ Cleanup error: {e}")
        logging.error(f"Cleanup error: {e}")
    print(f"✓ Cleanup complete. Deleted {deleted_count} files.")
    logging.info(f"Cleanup complete. Deleted {deleted_count} files.")
    return deleted_count > 0

def recorder():
    global file_counter, last_cleanup, packet_seq_num
    print(f"Recording from default microphone")
    logging.info("Recording from default microphone")
    while True:
        try:
            if check_disk_space():
                cleanup_old_files()
            # Modified: Handle queue overflow by dropping oldest item
            if analyze_queue.full():
                print(f"Analyze queue full ({analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}). Dropping oldest item.")
                logging.warning(f"Analyze queue full. Dropping oldest item.")
                try:
                    analyze_queue.get_nowait()
                except queue.Empty:
                    pass
            record_start_time = get_timestamp()
            send_record_alert(record_start_time)
            filename = f"Rec{file_counter:06d}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            print(f"\n Recording: {filename}")
            logging.info(f"Recording: {filename}")
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                device=MIC_DEVICE
            )
            sd.wait()
            wav.write(filepath, SAMPLE_RATE, audio.flatten())
            print(f"Saved: {filename}")
            logging.info(f"Saved: {filename}")
            analyze_queue.put((filepath, record_start_time, packet_seq_num))
            file_counter += 1
            packet_seq_num = (packet_seq_num + 1) % 65536
            if time.time() - last_cleanup > CLEAN_INTERVAL_DAYS * 86400:
                threading.Thread(target=cleanup_old_files, daemon=True).start()
                last_cleanup = time.time()
        except Exception as e:
            print(f"✗ Recording error: {e}")
            logging.error(f"Recording error: {e}")
            time.sleep(1)

def analyzer():
    pipeline = FeatureEngineeringPipeline()
    pipeline.add_block(Decimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(FrequencyPeakFinder(buffer_seconds=RECORD_SECONDS))
    pipeline.add_output_block(RPM(events_per_crankshaft_cycle=EVENTS_PER_CYCLE))
    while True:
        try:
            filepath, record_timestamp, seq_num = analyze_queue.get()
            print(f"Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
            logging.info(f"Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
            sample_rate, data = wav.read(filepath)
            if len(data.shape) > 1:
                data = data.flatten()
            datum = Datum(audio_array=data, sample_rate=sample_rate)
            datum = pipeline.run(datum)
            success = send_uart_packet(record_timestamp, datum, seq_num)
            if success:
                rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
                freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
                print(f"   ⚡ RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
        except Exception as e:
            print(f"Analysis error: {e}")
            logging.error(f"Analysis error: {e}")

def main():
    print("Audio Processing System Starting...")
    logging.info("Audio Processing System Starting")
    print(f"Recordings directory: {AUDIO_DIR}")
    if not init_uart():
        print("System starting without UART connection")
    send_sync_packet()
    recorder_thread = threading.Thread(target=recorder, daemon=True)
    analyzer_thread = threading.Thread(target=analyzer, daemon=True)
    recorder_thread.start()
    analyzer_thread.start()
    print("System started!")
    logging.info("System started")
    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(300)
            send_sync_packet()
    except KeyboardInterrupt:
        print("\n Stopping system...")
        logging.info("Stopping system")
        if ser:
            ser.close()
        print("System stopped")
        logging.info("System stopped")

if __name__ == "__main__":
    main()
