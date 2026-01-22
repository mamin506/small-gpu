from typing import List  # Import List type for type hinting
from .logger import logger  # Import logger from local logger module

class Memory:  # Define Memory class
    def __init__(self, dut, addr_bits, data_bits, channels, name):  # Constructor with device under test and memory parameters
        self.dut = dut  # Store reference to device under test
        self.addr_bits = addr_bits  # Number of address bits
        self.data_bits = data_bits  # Number of data bits
        self.memory = [0] * (2**addr_bits)  # Initialize memory array with zeros, size 2^addr_bits
        self.channels = channels  # Number of memory channels
        self.name = name  # Name of the memory instance

        # Get references to read signals from dut using name
        self.mem_read_valid = getattr(dut, f"{name}_mem_read_valid")  # Read valid signal
        self.mem_read_address = getattr(dut, f"{name}_mem_read_address")  # Read address signal
        self.mem_read_ready = getattr(dut, f"{name}_mem_read_ready")  # Read ready signal
        self.mem_read_data = getattr(dut, f"{name}_mem_read_data")  # Read data signal

        # If not program memory, get write signals
        if name != "program":
            self.mem_write_valid = getattr(dut, f"{name}_mem_write_valid")  # Write valid signal
            self.mem_write_address = getattr(dut, f"{name}_mem_write_address")  # Write address signal
            self.mem_write_data = getattr(dut, f"{name}_mem_write_data")  # Write data signal
            self.mem_write_ready = getattr(dut, f"{name}_mem_write_ready")  # Write ready signal

    def run(self):  # Simulate one memory cycle
        # Extract read valid bits for each channel
        mem_read_valid = [
            int(str(self.mem_read_valid.value)[i:i+1], 2)  # Convert each bit to int
            for i in range(0, len(str(self.mem_read_valid.value)), 1)  # Iterate over each bit
        ]

        # Extract read addresses for each channel
        mem_read_address = [
            int(str(self.mem_read_address.value)[i:i+self.addr_bits], 2)  # Convert address bits to int
            for i in range(0, len(str(self.mem_read_address.value)), self.addr_bits)  # Step by addr_bits
        ]
        mem_read_ready = [0] * self.channels  # Initialize read ready flags
        mem_read_data = [0] * self.channels  # Initialize read data values

        # Process read requests for each channel
        for i in range(self.channels):
            if mem_read_valid[i] == 1:  # If read valid
                mem_read_data[i] = self.memory[mem_read_address[i]]  # Read data from memory
                mem_read_ready[i] = 1  # Set ready flag
            else:
                mem_read_ready[i] = 0  # Not ready

        # Pack read data and ready flags into dut signals
        self.mem_read_data.value = int(''.join(format(d, '0' + str(self.data_bits) + 'b') for d in mem_read_data), 2)  # Concatenate data bits and set value
        self.mem_read_ready.value = int(''.join(format(r, '01b') for r in mem_read_ready), 2)  # Concatenate ready bits and set value

        # If not program memory, process write requests
        if self.name != "program":
            # Extract write valid bits for each channel
            mem_write_valid = [
                int(str(self.mem_write_valid.value)[i:i+1], 2)  # Convert each bit to int
                for i in range(0, len(str(self.mem_write_valid.value)), 1)  # Iterate over each bit
            ]
            # Extract write addresses for each channel
            mem_write_address = [
                int(str(self.mem_write_address.value)[i:i+self.addr_bits], 2)  # Convert address bits to int
                for i in range(0, len(str(self.mem_write_address.value)), self.addr_bits)  # Step by addr_bits
            ]
            # Extract write data for each channel
            mem_write_data = [
                int(str(self.mem_write_data.value)[i:i+self.data_bits], 2)  # Convert data bits to int
                for i in range(0, len(str(self.mem_write_data.value)), self.data_bits)  # Step by data_bits
            ]
            mem_write_ready = [0] * self.channels  # Initialize write ready flags

            # Process write requests for each channel
            for i in range(self.channels):
                if mem_write_valid[i] == 1:  # If write valid
                    self.memory[mem_write_address[i]] = mem_write_data[i]  # Write data to memory
                    mem_write_ready[i] = 1  # Set ready flag
                else:
                    mem_write_ready[i] = 0  # Not ready

            # Pack write ready flags into dut signal
            self.mem_write_ready.value = int(''.join(format(w, '01b') for w in mem_write_ready), 2)  # Concatenate ready bits and set value

    def write(self, address, data):  # Write data to memory at address
        if address < len(self.memory):  # Check address bounds
            self.memory[address] = data  # Write data

    def load(self, rows: List[int]):  # Load list of data into memory
        for address, data in enumerate(rows):  # Iterate over rows
            self.write(address, data)  # Write each data to corresponding address

    def display(self, rows, decimal=True):  # Display memory contents
        logger.info("\n")  # Print newline
        logger.info(f"{self.name.upper()} MEMORY")  # Print memory name

        table_size = (8 * 2) + 3  # Calculate table width
        logger.info("+" + "-" * (table_size - 3) + "+")  # Print table top border

        header = "| Addr | Data "  # Table header
        logger.info(header + " " * (table_size - len(header) - 1) + "|")  # Print header row

        logger.info("+" + "-" * (table_size - 3) + "+")  # Print header bottom border
        for i, data in enumerate(self.memory):  # Iterate over memory
            if i < rows:  # Only display up to 'rows' entries
                if decimal:  # If decimal display
                    row = f"| {i:<4} | {data:<4}"  # Format row with decimal data
                    logger.info(row + " " * (table_size - len(row) - 1) + "|")  # Print row
                else:  # If binary display
                    data_bin = format(data, f'0{16}b')  # Format data as 16-bit binary
                    row = f"| {i:<4} | {data_bin} |"  # Format row with binary data
                    logger.info(row + " " * (table_size - len(row) - 1) + "|")  # Print row
        logger.info("+" + "-" * (table_size - 3) + "+")  # Print table bottom border