import mujoco

def main():
    print("Loading URDF and translating to MJCF...")
    urdf_path = "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_humanoid_description/urdf/robot.urdf"
    xml_out_path = "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_humanoid_description/urdf/robot.xml"
    
    # This loads the URDF and compiles it to native MuJoCo format in memory
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    # This saves the translated memory out as a new .xml file
    mujoco.mj_saveLastXML(xml_out_path, model)
    print(f"Success! Saved native MuJoCo file to: {xml_out_path}")

if __name__ == "__main__":
    main()