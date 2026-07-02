import os
import struct

# --- The Master Variables ---
input_file = "VOICE1.BIN"
output_dir = "VOICE1BIN"
signature = b'\x69\x00\x00\x00\x01\x00\x00\x00\x24\x00\x00\x00\x22\x56\x00\x00'
offset_shift = 16       

# The Two-Tier Buffer System you discovered
first_page_size = 34776       
repeating_page_size = 32760   
marker_size = 8         

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit()

with open(input_file, "rb") as f:
    data = f.read()

chunks = data.split(signature)
os.makedirs(output_dir, exist_ok=True)

print(f"Starting final extraction. Stitching two-tier page breaks...")

count = 0
for chunk in chunks[1:]:
    if len(chunk) < offset_shift:
        continue

    # Extract the sample count
    num_samples = struct.unpack('<I', chunk[4:8])[0]
    expected_data_size = (num_samples // 64) * 36
    
    # Stitch the streaming buffers together
    raw_audio = b''
    read_cursor = offset_shift
    bytes_remaining = expected_data_size
    
    # Logic Switch for the Two-Tier system
    is_first_page = True
    
    while bytes_remaining > 0:
        # Determine current page size based on tier
        current_page_size = first_page_size if is_first_page else repeating_page_size
        
        chunk_to_read = min(current_page_size, bytes_remaining)
        
        # Safety catch
        if read_cursor + chunk_to_read > len(chunk):
            chunk_to_read = len(chunk) - read_cursor
            bytes_remaining = chunk_to_read 
            
        raw_audio += chunk[read_cursor : read_cursor + chunk_to_read]
        
        read_cursor += chunk_to_read
        bytes_remaining -= chunk_to_read
        
        # If there is still audio left, skip the 8-byte zeroes
        if bytes_remaining > 0:
            read_cursor += marker_size
            
        # Flip the switch so all subsequent loops use 32,760
        is_first_page = False

    # Synthesize the Microsoft RIFF/WAV Header
    riff_size = 4 + (8 + 20) + (8 + 4) + (8 + expected_data_size)
    header = b'RIFF' + struct.pack('<I', riff_size) + b'WAVEfmt '
    header += struct.pack('<I', 20)          
    header += struct.pack('<H', 0x0069)      
    header += struct.pack('<H', 1)           
    header += struct.pack('<I', 22050)       
    header += struct.pack('<I', 12403)       
    header += struct.pack('<H', 36)          
    header += struct.pack('<H', 4)           
    header += struct.pack('<H', 2)           
    header += struct.pack('<H', 64)          
    header += b'fact' + struct.pack('<I', 4) + struct.pack('<I', num_samples)
    header += b'data' + struct.pack('<I', expected_data_size)

    # Save the file
    with open(os.path.join(output_dir, f"voice_{count:04d}.lwav"), "wb") as f:
        f.write(header + raw_audio)
    
    count += 1

print(f"\nSuccess! All distortion completely eliminated. Saved to '{output_dir}'.")