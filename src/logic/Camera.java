package logic;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public class Camera
{
    private VideoCapture videoCapture;

    public Camera(int width, int height, int cameraIndex)
    {
        videoCapture = new VideoCapture();
        if (videoCapture.open(cameraIndex)) {
            int fourcc = VideoWriter.fourcc('M', 'J', 'P', 'G');    // might be useful if video capture takes much time
            videoCapture.set(Videoio.CAP_PROP_FOURCC, fourcc);
            videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, width);
            videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, height);
        }
    }

    public VideoCapture getVideoCapture()
    {
        return videoCapture;
    }

    public BufferedImage getFrame(int width, int height)
    {
        BufferedImage image = null;
        Mat frame = new Mat();

        try {
            // read the current frame
            this.videoCapture.read(frame);

            // if the frame is not empty, process it
            if (!frame.empty()) {
                Size size = new Size(width, height);
                Imgproc.resize(frame, frame, size);
                image = matToBufferedImage(frame);
            }
        } catch (Exception e) {
            System.err.println("Exception during the image elaboration: " + e);
        }

        return image;
    }

    public static BufferedImage matToBufferedImage(Mat original)
    {
        BufferedImage image = null;
        int width = original.width();
        int height = original.height();
        int channels = original.channels();
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

    public void stop()
    {
        videoCapture.release();
    }
}
