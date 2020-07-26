/*
 * Copyright (c) 2020 CertifAI
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.certifai;

import ai.certifai.haar.HaarFaceDetector;
import org.bytedeco.javacv.*;
import org.bytedeco.javacv.Frame;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;

import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

/**
 * Face Detection with HaarCascade
 *
 * @author ChiaWei lim
 */
public class FrontCamFaceDetection {

    private FrameGrabber frameGrabber;
    private OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
    private volatile boolean running = false;

    private static final int imageWidth = 640;
    private static final int imageHeight = 360;

    private HaarFaceDetector faceDetector = new HaarFaceDetector();


    private static CanvasFrame canvas;

    /**
     * Starts the frame grabbers and then the frame processing.
     */
    public void start() {
        frameGrabber = new OpenCVFrameGrabber(0);

        //frameGrabber.setFormat("mp4");
        frameGrabber.setImageWidth(imageWidth);
        frameGrabber.setImageHeight(imageHeight);

        System.out.println("Starting frame grabber");
        try {
            frameGrabber.start();
            System.out.println("Started frame grabber");
        } catch (FrameGrabber.Exception e) {
            System.out.println("Error when initializing the frame grabber. " + e);

            throw new RuntimeException("Unable to start the FrameGrabber", e);
        }

        process();

        System.out.println("Stopped frame grabbing.");
    }

    /**
     * Private method which will be called to star frame grabbing and carry on processing the grabbed frames
     */
    private void process() {
        running = true;
        while (running) {
            try {
                // Here we grab frames from our camera
                final Frame frame = frameGrabber.grab();

                Map<Rect, Mat> detectedFaces = faceDetector.detect(frame);
                Mat mat = toMatConverter.convert(frame);

                detectedFaces.entrySet().forEach(rectMatEntry -> {

                    rectangle(mat, new Point(rectMatEntry.getKey().x(), rectMatEntry.getKey().y()),
                            new Point(rectMatEntry.getKey().width() + rectMatEntry.getKey().x(), rectMatEntry.getKey().height() + rectMatEntry.getKey().y()),
                            Scalar.RED, 2, CV_AA, 0);

                    int posX = Math.max(rectMatEntry.getKey().x() - 10, 0);
                    int posY = Math.max(rectMatEntry.getKey().y() - 10, 0);
                });

                Mat matFlip = new Mat();
                flip(mat, matFlip, 1);
                // Show the processed mat in UI
                canvas.showImage(toMatConverter.convert(matFlip));

            } catch (FrameGrabber.Exception e) {
                System.out.println("Error when grabbing the frame. " +  e);
            } catch (Exception e) {
                System.out.println("Unexpected error occurred while grabbing and processing a frame." + e);
            }
        }
    }

    /**
     * Stops and released resources attached to frame grabbing. Stops frame processing and,
     */
    public void stop() {
        running = false;
        try {
            System.out.println("Releasing and stopping FrameGrabber");
            frameGrabber.release();
            frameGrabber.stop();
        } catch (FrameGrabber.Exception e) {
            System.out.println("Error occurred when stopping the FrameGrabber. " + e);
        }

    }

    public static void main(String[] args) {

        canvas = new CanvasFrame("Haar Cascade");
        canvas.setCanvasSize(imageWidth, imageHeight);

        FrontCamFaceDetection app = new FrontCamFaceDetection();

        System.out.println("This example works with front camera");

        System.out.println("Starting detection");
        new Thread(app::start).start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Stopping detection");
            app.stop();
        }));

        try {
            Thread.currentThread().join();
        } catch (InterruptedException ignored) { }
    }
}