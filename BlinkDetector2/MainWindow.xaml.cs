using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using OpenCvSharp;
using DlibDotNet;
using System.Diagnostics;
using System.Runtime.InteropServices;


namespace BlinkDetector2
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        public MainWindow()
        {
            InitializeComponent();

            try
            {
                //capture
                var cap = new VideoCapture(0);

                if (!cap.IsOpened())
                {
                    //when unable connect to webcamera
                    Debug.WriteLine("Unable to connect to camera");
                    return;
                }

                using (var win = new ImageWindow())
                {
                    //Dlib 68 points facelandmark Detector
                    using (var poseModel = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
                    //opencv face detector
                    using (var detector = new CascadeClassifier("haarcascade_frontalface_alt2.xml"))
                    {
                        while (!win.IsClosed())
                        {
                            //opencv mat
                            var matVideCaptureImage = new Mat();
                            if (!cap.Read(matVideCaptureImage))
                            {
                                break;
                            }
                            // convert to gray scale
                            var matGrayscaleImage = new Mat();
                            Cv2.CvtColor(
                                src: matVideCaptureImage,
                                dst: matGrayscaleImage,
                                code: ColorConversionCodes.RGB2GRAY
                                );
                            //generate array
                            var array = new byte[matGrayscaleImage.Width * matGrayscaleImage.Height * matGrayscaleImage.ElemSize()];
                            //copy mat to array
                            Marshal.Copy(matGrayscaleImage.Data, array, 0, array.Length);
                            //load image
                            using (var cimg = Dlib.LoadImageData<BgrPixel>(array, (uint)matGrayscaleImage.Height, (uint)matGrayscaleImage.Width, (uint)(matGrayscaleImage.Width * matGrayscaleImage.ElemSize())))
                            {
                                //delect faces
                                var faces = detector.DetectMultiScale(matGrayscaleImage);
                                //var faces = detector.Operator(cimg);

                                var shapes = new List<FullObjectDetection>();
                                //for (var i = 0; i < faces.Length; i++)
                                //{
                                //    var det = poseModel.Detect(cimg, faces[i]);
                                //    shapes.Add(det);
                                //}

                                //face count more than 0
                                if (faces.Length > 0)
                                {
                                    foreach (var face in faces)
                                    {
                                        //face for each
                                        var tmpFace = new DlibDotNet.Rectangle(face.X, face.Y, face.X + face.Width, face.Y + face.Height);
                                        
                                        //point detect
                                        var det = poseModel.Detect(cimg, tmpFace);

                                        for(var i= 0; i < det.Parts; i++)
                                        {
                                            //ポイントの一部抽出
                                            var point = det.GetPart((uint)i);
                                        }
                                        shapes.Add(det);
                                    }
                                }

                                win.ClearOverlay();
                                win.SetImage(cimg);
                                var lines = Dlib.RenderFaceDetections(shapes);
                                win.AddOverlay(lines);

                                foreach (var line in lines)
                                    line.Dispose();


                            }
                        }
                    }
                }
            } catch(Exception ex)
            {
                Debug.WriteLine(ex.Message);
            }
        }
    }
}
