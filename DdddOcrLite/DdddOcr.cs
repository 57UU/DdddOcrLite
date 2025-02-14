using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace DdddOcrLite;

public partial class DdddOcr:IDisposable
{
    private InferenceSession session;
    private RunOptions runOptions;
    public DdddOcr(string modelPath="models/common_old.onnx")
    {
        session = new InferenceSession(modelPath);
        runOptions = new RunOptions();
    }
    public string Classification(Image<Rgb24> image)
    {
        // 调整图像大小
        var options = new ResizeOptions
        {
            Size = new Size((int)(image.Width*64.0f/image.Height), 64),
            Mode=ResizeMode.Stretch
        };
        image.Mutate(ctx =>ctx.Resize(options));

        using var gray = image.CloneAs<L8>();

        DenseTensor<float> processedImage = new([1, gray.Height, gray.Width]);

        gray.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var pixelSpan = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    processedImage[0, y, x] = ((pixelSpan[x].PackedValue / 255f) - 0.5f) *2;
                }
            }
        });

        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(
            OrtMemoryInfo.DefaultInstance,
            processedImage.Buffer,
            [1,1, gray.Height, gray.Width]
            );

        var inputs = new Dictionary<string, OrtValue>
            {
                { "input1", inputOrtValue }
            };

        using var results = session.Run(runOptions, inputs, session.OutputNames);

        var output = results[0].GetTensorDataAsSpan<float>();

        var argmaxResult= ArgMax(output,charsetLength);

        int lastItem = 0;
        StringBuilder sb= new StringBuilder();
        foreach (var item in argmaxResult)
        {
            if (item == lastItem)
            {
                continue;
            }
            else
            {
                lastItem = item;
            }
            if (lastItem != 0)
            {
                sb.Append(charset[item]);
            }
        }
        return sb.ToString();
    }
    private static int[] ArgMax(ReadOnlySpan<float> data,int segmentLength)
    {
        int count=data.Length/segmentLength;
        int[] result=new int[count];
        for (int i = 0; i < count; i++) {
            int offset=i*segmentLength;
            result[i]=MaxIndex(data, offset,segmentLength);
        }
        return result;
    }
    private static int MaxIndex(ReadOnlySpan<float> data, int offset, int length)
    {
        int index = 0;
        for (int i = 0; i < length; i++)
        {
            if (data[offset+i] > data[offset + index])
            {
                index = i;
            }
        }
        return index;
    }

    public void Dispose()
    {
        session.Dispose();
        runOptions.Dispose();
    }
}
