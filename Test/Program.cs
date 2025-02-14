

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

var ocr = new DdddOcrLite.DdddOcr();
var image = Image.Load<Rgb24>("1.png");
Console.WriteLine(ocr.Classification(image));