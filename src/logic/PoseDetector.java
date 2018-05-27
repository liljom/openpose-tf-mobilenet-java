package logic;

import logic.tensorflow.TensorFlowInferenceInterface;

import java.nio.file.Paths;
import java.util.Vector;

public class PoseDetector
{
    private final String MODEL_FILE = "resources/thin.pb";
    private final String INPUT_NAME = "image";
    private final String OUTPUT_NAME = "Openpose/concat_stage7";
    private final float NMS_Threshold = 0.15f;
    private final float Local_PAF_Threshold = 0.2f;
    private final float Part_Score_Threshold = 0.2f;
    private final int PAF_Count_Threshold = 5;
    private final int Part_Count_Threshold = 4;
    private int MapHeight;
    private int MapWidth;
    private final int HeatMapCount = 19;
    private final int MaxPairCount = 17;
    private final int PafMapCount = 38;
    private final int MaximumFilterSize = 5;
    public static final int[][] CocoPairs = {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11},
            {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};
    private final int[][] CocoPairsNetwork = {{12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1}, {2, 3},
            {4, 5}, {6, 7}, {8, 9}, {10, 11}, {28, 29}, {30, 31}, {34, 35}, {32, 33}, {36, 37}, {18, 19}, {26, 27}};

    private int[] colorChannels;
    private String[] outputNames;
    private float[] float_image;
    private float[] output_tensor;
    private TensorFlowInferenceInterface inferenceInterface;
    private int inWidth;
    private int inHeight;

    public PoseDetector(String colorChannels, int width, int height)
    {
        inWidth = width;
        inHeight = height;
        MapWidth = width / 8;
        MapHeight = height / 8;
        inferenceInterface = new TensorFlowInferenceInterface(Paths.get(MODEL_FILE));
        outputNames = new String[]{OUTPUT_NAME};
        float_image = new float[width * height * 3];
        output_tensor = new float[MapHeight * MapWidth * (HeatMapCount + PafMapCount)];
        if (colorChannels.equals("BGR")) {
            this.colorChannels = new int[]{0, 1, 2};
        } else if (colorChannels.equals("RGB")) {
            this.colorChannels = new int[]{2, 1, 0};
        }
    }

    public void setImage(int[] rgbImage)
    {
        if (rgbImage.length != inWidth * inHeight) {
            System.err.println("input size doesn't match (" + inWidth + "x" + inHeight + ")");
        } else {
            System.out.println("input size (" + inWidth + "x" + inHeight + ")");
            for (int i = 0; i < rgbImage.length; ++i) {
                final int val = rgbImage[i];
                float_image[i * 3 + colorChannels[0]] = ((val >> 16) & 0xFF); //R
                float_image[i * 3 + colorChannels[1]] = ((val >> 8) & 0xFF);  //G
                float_image[i * 3 + colorChannels[2]] = (val & 0xFF);         //B
            }
        }
    }

    public Vector<Human> recognizePoses()
    {

        inferenceInterface.feed(INPUT_NAME, float_image, 1, inHeight, inWidth, 3);

        inferenceInterface.run(outputNames);

        inferenceInterface.fetch(OUTPUT_NAME, output_tensor);

        Vector<int[]> coordinates[] = new Vector[HeatMapCount - 1];

        // eliminate duplicate part recognitions
        for (int i = 0; i < (HeatMapCount - 1); i++) {
            coordinates[i] = new Vector<int[]>();
            for (int j = 0; j < MapHeight; j++) {
                for (int k = 0; k < MapWidth; k++) {
                    int[] coordinate = {j, k};
                    float max_value = 0;
                    for (int dj = -(MaximumFilterSize - 1) / 2; dj < (MaximumFilterSize + 1) / 2; dj++) {
                        if ((j + dj) >= MapHeight || (j + dj) < 0) {
                            break;
                        }
                        for (int dk = -(MaximumFilterSize - 1) / 2; dk < (MaximumFilterSize + 1) / 2; dk++) {
                            if ((k + dk) >= MapWidth || (k + dk) < 0) {
                                break;
                            }
                            float value = output_tensor[(HeatMapCount + PafMapCount) * MapWidth * (j + dj) + (HeatMapCount + PafMapCount) * (k + dk) + i];
                            if (value > max_value) {
                                max_value = value;
                            }
                        }
                    }
                    if (max_value > NMS_Threshold) {
                        if (max_value == output_tensor[(HeatMapCount + PafMapCount) * MapWidth * j + (HeatMapCount + PafMapCount) * k + i]) {
                            coordinates[i].addElement(coordinate);
                        }
                    }
                }
            }
        }

        // eliminate duplicate connections
        Vector<int[]> pairs[] = new Vector[MaxPairCount];
        Vector<int[]> pairs_final[] = new Vector[MaxPairCount];
        Vector<Float> pairs_scores[] = new Vector[MaxPairCount];
        Vector<Float> pairs_scores_final[] = new Vector[MaxPairCount];
        for (int i = 0; i < MaxPairCount; i++) {
            pairs[i] = new Vector<int[]>();
            pairs_scores[i] = new Vector<Float>();
            pairs_final[i] = new Vector<int[]>();
            pairs_scores_final[i] = new Vector<Float>();
            Vector<Integer> part_set = new Vector<Integer>();
            for (int p1 = 0; p1 < coordinates[CocoPairs[i][0]].size(); p1++) {
                for (int p2 = 0; p2 < coordinates[CocoPairs[i][1]].size(); p2++) {
                    int count = 0;
                    float score = 0.0f;
                    float scores[] = new float[10];
                    int p1x = coordinates[CocoPairs[i][0]].get(p1)[0];
                    int p1y = coordinates[CocoPairs[i][0]].get(p1)[1];
                    int p2x = coordinates[CocoPairs[i][1]].get(p2)[0];
                    int p2y = coordinates[CocoPairs[i][1]].get(p2)[1];
                    float dx = p2x - p1x;
                    float dy = p2y - p1y;
                    float normVec = (float) Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));

                    if (normVec < 0.0001f) {
                        break;
                    }
                    float vx = dx / normVec;
                    float vy = dy / normVec;
                    for (int t = 0; t < 10; t++) {
                        int tx = (int) ((float) p1x + (t * dx / 9) + 0.5);
                        int ty = (int) ((float) p1y + (t * dy / 9) + 0.5);
                        int location = tx * (HeatMapCount + PafMapCount) * MapWidth + ty * (HeatMapCount + PafMapCount) + HeatMapCount;
                        scores[t] = vy * output_tensor[location + CocoPairsNetwork[i][0]];
                        scores[t] += vx * output_tensor[location + CocoPairsNetwork[i][1]];
                    }
                    for (int h = 0; h < 10; h++) {
                        if (scores[h] > Local_PAF_Threshold) {
                            count += 1;
                            score += scores[h];
                        }
                    }
                    if (score > Part_Score_Threshold && count >= PAF_Count_Threshold) {
                        boolean inserted = false;
                        int pair[] = {p1, p2};
                        for (int l = 0; l < pairs[i].size(); l++) {
                            if (score > pairs_scores[i].get(l)) {
                                pairs[i].insertElementAt(pair, l);
                                pairs_scores[i].insertElementAt(score, l);
                                inserted = true;
                                break;
                            }
                        }
                        if (!inserted) {
                            pairs[i].addElement(pair);
                            pairs_scores[i].addElement(score);
                        }
                    }
                }
            }
            for (int m = 0; m < pairs[i].size(); m++) {
                boolean conflict = false;
                for (int n = 0; n < part_set.size(); n++) {
                    if (pairs[i].get(m)[0] == part_set.get(n) || pairs[i].get(m)[1] == part_set.get(n)) {
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) {
                    pairs_final[i].addElement(pairs[i].get(m));
                    pairs_scores_final[i].addElement(pairs_scores[i].get(m));
                    part_set.addElement(pairs[i].get(m)[0]);
                    part_set.addElement(pairs[i].get(m)[1]);
                }
            }
        }

        Vector<Human> humans = new Vector<Human>();
        Vector<Human> humans_final = new Vector<Human>();
        for (int i = 0; i < MaxPairCount; i++) {
            for (int j = 0; j < pairs_final[i].size(); j++) {
                boolean merged = false;
                int p1 = CocoPairs[i][0];
                int p2 = CocoPairs[i][1];
                int ip1 = pairs_final[i].get(j)[0];
                int ip2 = pairs_final[i].get(j)[1];
                for (int k = 0; k < humans.size(); k++) {
                    Human human = humans.get(k);
                    if ((ip1 == human.coords_index_set[p1] && human.coords_index_assigned[p1]) || (ip2 == human.coords_index_set[p2] && human.coords_index_assigned[p2])) {
                        human.parts_coords[p1] = coordinates[p1].get(ip1);
                        human.parts_coords[p2] = coordinates[p2].get(ip2);
                        human.coords_index_set[p1] = ip1;
                        human.coords_index_set[p2] = ip2;
                        human.coords_index_assigned[p1] = true;
                        human.coords_index_assigned[p2] = true;
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    Human human = new Human();
                    human.parts_coords[p1] = coordinates[p1].get(ip1);
                    human.parts_coords[p2] = coordinates[p2].get(ip2);
                    human.coords_index_set[p1] = ip1;
                    human.coords_index_set[p2] = ip2;
                    human.coords_index_assigned[p1] = true;
                    human.coords_index_assigned[p2] = true;
                    humans.addElement(human);
                }
            }
        }

        // remove people with too few parts
        for (int i = 0; i < humans.size(); i++) {
            int human_part_count = 0;
            for (int j = 0; j < HeatMapCount - 1; j++) {
                if (humans.get(i).coords_index_assigned[j]) {
                    human_part_count += 1;
                }
            }
            if (human_part_count > Part_Count_Threshold) {
                humans_final.addElement(humans.get(i));
            }
        }

        System.out.println("Number of humans on the screen: " + humans_final.size());

        return humans_final;
    }
}
