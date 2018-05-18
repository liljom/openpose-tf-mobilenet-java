import gui.MainSceneController;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import org.opencv.core.Core;

import java.io.IOException;

public class App extends Application
{
    public static void main(String[] args)
    {
        // load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        launch(args);
    }

    @Override
    public void start(Stage stage) throws IOException
    {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("gui/MainScene.fxml"));
        Pane root = loader.load();
        Scene scene = new Scene(root);
        stage.setTitle("Pose detection");
        stage.setScene(scene);
        stage.show();

        MainSceneController controller = loader.getController();
        controller.init();

        stage.setOnCloseRequest((event -> controller.stopAcquisition()));
    }
}
