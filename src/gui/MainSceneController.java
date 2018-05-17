package gui;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import logic.Human;
import logic.PoseDetector;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class MainSceneController {
    @FXML
    private ImageView frame;

    @FXML
    private Button btnStart;

    private ScheduledExecutorService timer;
    private VideoCapture videoCapture;
    private boolean cameraActive = false;
    private long time = System.currentTimeMillis();

    private PoseDetector detector;
    private int inWidth = 488;
    private int inHeight = 272;

    public void init() {
        this.videoCapture = new VideoCapture();

        detector = new PoseDetector("RGB", inWidth, inHeight);

        frame.setPreserveRatio(true);
    }

    @FXML
    void start() {
        if (!this.cameraActive) {
            this.videoCapture.open(0);

            // is the video stream available?
            if (this.videoCapture.isOpened()) {
                this.cameraActive = true;
//				int fourcc = VideoWriter.fourcc('M', 'J', 'P', 'G');    // might be useful if video capture takes much time
//				videoCapture.set(Videoio.CAP_PROP_FOURCC, fourcc);
                videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 1920);
                videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 1080);

                // grab a frame every minimum 33 ms (max 30 frames/sec)
                Runnable frameGrabber = () -> {

                    BufferedImage imgOriginal = getFrame(inWidth, inHeight);
                    BufferedImage imgToRecognition = new BufferedImage(imgOriginal.getWidth(), imgOriginal.getHeight(), BufferedImage.TYPE_INT_BGR);
                    imgToRecognition.getGraphics().drawImage(imgOriginal, 0, 0, null);
                    int[] intValues = ((DataBufferInt) imgToRecognition.getRaster().getDataBuffer()).getData();
                    detector.setImage(intValues);

                    long time1 = System.currentTimeMillis();
                    List<Human> recognitions = detector.recognizePoses();
                    System.out.println("processing took " + (System.currentTimeMillis() - time1) + " ms");

                    Graphics2D canvas = imgOriginal.createGraphics();
                    long dTime = System.currentTimeMillis() - time;
                    System.out.println("whole cycle took " + dTime);
                    time = System.currentTimeMillis();

                    for (Human human : recognitions) {
                        for (int i = 0; i < 18; i++) {
                            int x = human.parts_coords[i][1] * 8;
                            int y = human.parts_coords[i][0] * 8;

                            Stroke stroke = canvas.getStroke();
                            canvas.setStroke(new BasicStroke(2));
                            canvas.setColor(Color.green);
                            canvas.drawOval(x, y, 6, 6);
                            canvas.drawString(Human.parts.get(i), x - 2, y - 2);
                            double fps = Math.round(10000.0 / dTime) / 10.0;
                            canvas.drawString(fps + " fps", 20, 20);
                            canvas.setStroke(stroke);
                        }
                    }

                    Image im = SwingFXUtils.toFXImage(imgOriginal, null);

                    frame.setImage(im);
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                this.btnStart.setText("Stop");
            } else {
                System.err.println("Failed to open the camera connection...");
            }
        } else {
            this.cameraActive = false;
            this.btnStart.setText("Start");
            this.stopAcquisition();
        }
    }

    private BufferedImage getFrame(int width, int height) {
        BufferedImage image = null;
        Mat frame = new Mat();

        // check if the capture is open
        if (this.videoCapture.isOpened()) {
            try {
                // read the current frame
                this.videoCapture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    System.out.println("original video input size: " + frame.width() + "x" + frame.height());
                    Size size = new Size(width, height);
                    Imgproc.resize(frame, frame, size);
                    image = matToBufferedImage(frame);
                }

            } catch (Exception e) {
                System.err.println("Exception during the image elaboration: " + e);
            }
        }

        return image;
    }

    public static BufferedImage matToBufferedImage(Mat original) {
        // init
        BufferedImage image = null;
        int width = original.width(), height = original.height(), channels = original.channels();
        byte[] sourcePixels = new byte[width * height * channels];
        original.get(0, 0, sourcePixels);

        if (original.channels() > 1) {
            image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        } else {
            image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        }
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);

        return image;
    }

    public void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }

        if (this.videoCapture.isOpened()) {
            // release the camera
            this.videoCapture.release();
        }
    }
}
