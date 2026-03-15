import os
import sys
import glob
import subprocess
from cocotb_tools.runner import get_runner

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test"))

def test_gpu_runner():
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    sv2v_bin = os.path.join(proj_dir, "tools/sv2v/bin/sv2v")
    sv_dir = os.path.join(proj_dir, "gpu_files")
    v_output = os.path.join(proj_dir, "gpu_files/built_gpu.v")

    sv_files = glob.glob(os.path.join(sv_dir, "*.sv"))

    print(f"--- Converting {len(sv_files)} SystemVerilog files to Verilog ---")
    try:
        cmd = [sv2v_bin] + sv_files + ["-w", v_output]
        subprocess.run(cmd, check=True)
        # Prepend timescale directive for Icarus Verilog
        with open(v_output, 'r') as f:
            content = f.read()
        with open(v_output, 'w') as f:
            f.write("`timescale 1us/1ns\n" + content)
    except subprocess.CalledProcessError as e:
        print(f"sv2v conversion failed: {e}")
        return

    sim = os.getenv("SIM", "icarus")
    runner = get_runner(sim)
    
    runner.build(
        sources=[v_output],
        hdl_toplevel="gpu",
    )

    runner.test(
        hdl_toplevel="gpu",
        test_module="test_matadd",
        test_dir=os.path.join(proj_dir, "test")
    )

if __name__ == "__main__":
    test_gpu_runner()
