from typing import List  # Import List type for type hinting
import cocotb  # Import cocotb for coroutine-based testbench
from cocotb.clock import Clock  # Import Clock class to generate clock signal
from cocotb.triggers import RisingEdge  # Import RisingEdge trigger for clock edge synchronization
from .memory import Memory  # Import custom Memory class from local module

async def setup(
    dut,  # Device Under Test (DUT) object
    program_memory: Memory,  # Program memory interface
    program: List[int],  # List of instructions to load into program memory
    data_memory: Memory,  # Data memory interface
    data: List[int],  # List of data to load into data memory
    threads: int  # Number of threads to configure
):
    # Setup Clock
    clock = Clock(dut.clk, 25, units="us")  # Create a clock with 25 microsecond period on dut.clk
    cocotb.start_soon(clock.start())  # Start the clock asynchronously

    # Reset
    dut.reset.value = 1  # Assert reset signal
    await RisingEdge(dut.clk)  # Wait for rising edge of clock
    dut.reset.value = 0  # Deassert reset signal

    # Load Program Memory
    program_memory.load(program)  # Load instructions into program memory

    # Load Data Memory
    data_memory.load(data)  # Load data into data memory

    # Device Control Register
    dut.device_control_write_enable.value = 1  # Enable write to device control register
    dut.device_control_data.value = threads  # Set number of threads in device control register
    await RisingEdge(dut.clk)  # Wait for rising edge of clock
    dut.device_control_write_enable.value = 0  # Disable write to device control register

    # Start
    dut.start.value = 1  # Assert start signal to begin operation
