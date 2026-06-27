curl -X POST http://localhost:5000/activate_gripper # Activate gripper
curl -X POST http://localhost:5000/close_gripper # Close gripper
curl -X POST http://localhost:5000/open_gripper # Open gripper
curl -X POST http://localhost:5000/getpos # Print current end-effector pose in xyz translation and xyzw quaternions
curl -X POST http://localhost:5000/getpos_euler # Get current end-effector pose in xyz translation and xyz euler angles
curl -X POST http://localhost:5000/jointreset # Perform joint reset
curl -X POST http://localhost:5000/stopimp # Stop the impedance controller
curl -X POST http://localhost:5000/startimp # Start the impedance controller (**Only run this after stopimp**)
