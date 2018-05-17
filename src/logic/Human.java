package logic;

import java.util.Arrays;
import java.util.List;

public class Human {
    public static List<String> parts = Arrays.asList("nose", "neck", "rShoulder", "rElbow", "rWist", "lShoulder", "lElbow",
            "lWrist", "rHip", "rKnee", "rAnkle", "lHip", "lKnee", "lAnkle", "rEye", "lEye", "rEar", "lEar");
    public int parts_coords[][] = new int[18][2];
    public int coords_index_set[] = new int[18];
    public boolean coords_index_assigned[] = new boolean[18];
}
