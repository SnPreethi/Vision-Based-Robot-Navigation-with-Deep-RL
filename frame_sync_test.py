def test_frame_sync(env):
    # Add to your main training script BEFORE the training loop
    # Call it after creating the environment:
    # env = RosVisionEnv(...)
# test_frame_sync(env)
    """Test that we're getting fresh frames after each action."""
    print("\nTesting frame synchronization...")
    env.reset()
    frame_ids = []
    
    for i in range(10):
        action = np.array([0.5, 0.0])  # Move forward
        obs, reward, done, truncated, info = env.step(action)
        
        frame_id = info.get('frame_id', -1)
        frame_ids.append(frame_id)
        print(f"  Step {i}: frame_id={frame_id}, wait={info.get('frame_wait_ms', 0)}ms")
    
    # Check that frame_ids are strictly increasing
    is_increasing = all(frame_ids[i] < frame_ids[i+1] for i in range(len(frame_ids)-1))
    
    if is_increasing:
        print("Frame sync test PASSED - all frames are fresh!")
    else:
        print("Frame sync test FAILED - some frames are stale!")
        print(f"Frame IDs: {frame_ids}") 
    env.reset()
    print()