package gui;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import logic.Camera;
import logic.Human;
import logic.PoseDetector;

import java.awt.*;
import java.awt.geom.Line2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class MainSceneController
{
    @FXML
    private ImageView frame;

    @FXML
    private Button btnStart;

    private Camera camera;
    private PoseDetector detector;
    private ScheduledExecutorService timer;
    private boolean cameraActive;
    private long time = System.currentTimeMillis();

    private final int inWidth = 512;
    private final int inHeight = 288;

    public void init()
    {
        detector = new PoseDetector("BGR", inWidth, inHeight);
    }

    @FXML
    void start()
    {
        if (!this.cameraActive)
        {

            camera = new Camera(1920, 1080, 0);

            if (camera.getVideoCapture().isOpened())
            {
                this.cameraActive = true;

                // grab a frame every minimum 33 ms (max 30 frames/sec)
                Runnable frameGrabber = () -> {

                    BufferedImage imgOriginal = camera.getFrame(inWidth, inHeight);
                    int[] intValues = getImageData(imgOriginal);
                    detector.setImage(intValues);

                    long time1 = System.currentTimeMillis();
                    List<Human> recognitions = detector.recognizePoses();
                    System.out.println("processing took " + (System.currentTimeMillis() - time1) + " ms");

                    displayResults(imgOriginal, recognitions);
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                this.btnStart.setText("Stop");
            }
            else
            {
                System.err.println("Failed to open the camera connection...");
            }
        }
        else
        {
            this.cameraActive = false;
            camera.stop();
            this.btnStart.setText("Start");
            this.stopAcquisition();
        }
    }

    private int[] getImageData(BufferedImage image)
    {
        BufferedImage imgToRecognition = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_BGR);
        imgToRecognition.getGraphics().drawImage(image, 0, 0, null);
        int[] data = ((DataBufferInt) imgToRecognition.getRaster().getDataBuffer()).getData();
        return data;
    }

    private void displayResults(BufferedImage imgOriginal, List<Human> recognitions)
    {
        Graphics2D canvas = imgOriginal.createGraphics();
        long dTime = System.currentTimeMillis() - time;
        System.out.println("whole cycle took " + dTime);
        time = System.currentTimeMillis();

        Stroke stroke = canvas.getStroke();
        canvas.setStroke(new BasicStroke(2));
        canvas.setColor(Color.green);

        int[][] pairs = PoseDetector.CocoPairs;

        for (Human human : recognitions)
        {
            for (int i = 0; i < 18; i++)
            {
                int x = human.parts_coords[i][1] * 8;
                int y = human.parts_coords[i][0] * 8;

                if (x != 0 && y !=0)
                {
                    canvas.drawOval(x, y, 6, 6);
//                    canvas.drawString(Human.parts.get(i), x - 4, y - 4);

                    for (int j = 0; j < pairs.length; j++)
                    {
                        if (pairs[j][0] == i)
                        {
                            int v = human.parts_coords[pairs[j][1]][1] * 8;
                            int w = human.parts_coords[pairs[j][1]][0] * 8;
                            if (v != 0 && w != 0)
                            {
                                canvas.draw(new Line2D.Double(x, y, v, w));
                            }
                        }
                    }
                }
            }
        }
        canvas.setStroke(stroke);
        double fps = Math.round(10000.0 / dTime) / 10.0;
        canvas.drawString(fps + " fps", 20, 20);

        Image im = SwingFXUtils.toFXImage(imgOriginal, null);
        frame.setImage(im);
    }

    public void stopAcquisition()
    {
        if (this.timer != null && !this.timer.isShutdown())
        {
            try
            {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e)
            {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }
        camera.stop();
    }
}
