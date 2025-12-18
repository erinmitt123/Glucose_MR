using System.Text;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;
using Color = Unity.XR.PXR.SecureMR.Color;

namespace PicoXR.SecureMR.Demo
{

    public class MinimalApp2 : MonoBehaviour
    {
        public TextAsset helmetGltfAsset;
        public int vstWidth = 1024;
        public int vstHeight = 1024;

        private Provider provider;
        private Pipeline pipeline;
        private Tensor gltfTensor;
        private Tensor gltfPlaceholderTensor;

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        private void Start()
        {
            CreateProvider();
            CreateGlobals();
            CreatePipeline();
        }

        private void Update()
        {
            RunPipeline();
        }

        private void CreateProvider()
        {
            provider = new Provider(vstWidth, vstHeight);
        }

        private void CreateGlobals()
        {
            // Create GLTF tensor
            gltfTensor = provider.CreateTensor<Gltf>(helmetGltfAsset.bytes);

        }

        private void CreatePipeline()
        {
            pipeline = provider.CreatePipeline();

            // Create transform matrix tensor
            int[] transformDim = { 4, 4 };
            var transformShape = new TensorShape(transformDim);
            float[] transformData =
            {
                0.5f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.5f, 0.0f, -0.25f,
                0.0f, 0.0f, 0.5f, -6.5f,
                0.0f, 0.0f, 0.0f, 1.0f
            };
            var poseTensor = pipeline.CreateTensor<float, Matrix>(1, transformShape, transformData);

            // Create GLTF tensor placeholder
            gltfPlaceholderTensor = pipeline.CreateTensorReference<Gltf>();

            // Create render GLTF operator
            var renderGltfOperator = pipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            renderGltfOperator.SetOperand("gltf", gltfPlaceholderTensor);
            renderGltfOperator.SetOperand("world pose", poseTensor);
            
            RenderTextOperatorConfiguration renderTextConfiguration = new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif,"en-US",1440,960);
            var renderTextOp = pipeline.CreateOperator<RenderTextOperator>(renderTextConfiguration);
            //var textTensor = pipeline.CreateTensor<sbyte,Scalar>(1, new TensorShape(30));
            var textTensor = pipeline.CreateTensor<byte,Scalar>(1, new TensorShape(30), Encoding.UTF8.GetBytes("Hello World"));
            renderTextOp.SetOperand("text", textTensor);
            //var startPositionTensor = pipeline.CreateTensor<float,Point>(2, new TensorShape(1));
            var startPositionTensor = pipeline.CreateTensor<float,Point>(2, new TensorShape(1),new float[] { 0.1f, 0.3f});
            renderTextOp.SetOperand("start", startPositionTensor);
            renderTextOp.SetOperand("gltf", gltfPlaceholderTensor);
            var colorsTensor = pipeline.CreateTensor<byte,Color>(4, new TensorShape(2), new byte[]{255, 255, 255, 255, 0, 0, 0, 255});
            renderTextOp.SetOperand("colors", colorsTensor);
            var textureIDTensor = pipeline.CreateTensor<ushort,Scalar>(1, new TensorShape(1), new ushort[] { 0});
            renderTextOp.SetOperand("texture ID", textureIDTensor);
            var fontSizeTensor = pipeline.CreateTensor<float,Scalar>(1, new TensorShape(1), new float[] { 144f });
            renderTextOp.SetOperand("font size", fontSizeTensor);
            
            
            //textTensor.Reset(Encoding.UTF8.GetBytes("Hello World"));
            //startPositionTensor.Reset(new float[] { 0.1f, 0.3f});
            //colorsTensor.Reset(new byte[]{255, 255, 255, 255, 0, 0, 0, 255});
            //textureIDTensor.Reset(new ushort[] { 0});
            //fontSizeTensor.Reset(new float[] { 144f });
            
        }

        private void RunPipeline()
        {
            Debug.Log("Running pipeline...");

            var tensorMapping = new TensorMapping();

            tensorMapping.Set(gltfPlaceholderTensor, gltfTensor);

            pipeline.Execute(tensorMapping);
        }

    }
}