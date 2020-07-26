package ai.certifai.haar;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.CvMemStorage;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.cvReleaseMemStorage;

/**
 * Face detector using haar classifier cascades
 *
 * @author Chia Wei Lim
 */
public class HaarFaceDetector {

    //private CvHaarClassifierCascade haarClassifierCascade;
    CascadeClassifier faceCascade;
    private CvMemStorage storage;
    private OpenCVFrameConverter.ToIplImage iplImageConverter;
    private OpenCVFrameConverter.ToMat toMatConverter;

    public HaarFaceDetector() {
        iplImageConverter = new OpenCVFrameConverter.ToIplImage();
        toMatConverter = new OpenCVFrameConverter.ToMat();

        try {
            File haarCascade = new File(this.getClass().getClassLoader().getResource("haarcascade_frontalface_alt.xml").toURI());
            System.out.println("Haar Cascade file located at : " + haarCascade.getAbsolutePath());
            //haarClassifierCascade = new CvHaarClassifierCascade(cvload(haarCascade.getAbsolutePath()));
            faceCascade = new CascadeClassifier(haarCascade.getCanonicalPath());

        } catch (Exception e) {
            System.out.println("Error when trying to get the haar cascade. " + e);
            throw new IllegalStateException("Error when trying to get the haar cascade", e);
        }
        storage = CvMemStorage.create();
    }

    /**
     * Detects and returns a map of cropped faces from a given captured frame
     *
     * @param frame the frame captured by the {@link org.bytedeco.javacv.FrameGrabber}
     * @return A map of faces along with their coordinates in the frame
     */
    public Map<Rect, Mat> detect(Frame frame) {
        Map<Rect, Mat> detectedFaces = new HashMap<>();

        /*
         * return a CV Sequence (kind of a list) with coordinates of rectangle face area.
         * (returns coordinates of left top corner & right bottom corner)
         */
        RectVector detectObjects = new RectVector();

        Mat matImage = toMatConverter.convert(frame);
        faceCascade.detectMultiScale(matImage, detectObjects);

        long numberOfPeople = detectObjects.size();
        for (int i = 0; i < numberOfPeople; i++) {
            Rect rect = detectObjects.get(i);
            Mat croppedMat = matImage.apply(new Rect(rect.x(), rect.y(), rect.width(), rect.height()));
            detectedFaces.put(rect, croppedMat);
        }

        return detectedFaces;
    }

    @Override
    public void finalize() {
        cvReleaseMemStorage(storage);
    }
}