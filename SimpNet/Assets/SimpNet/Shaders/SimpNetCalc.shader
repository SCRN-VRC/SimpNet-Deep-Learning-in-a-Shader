Shader "SimpNet/SimpNetCalc"
{
    Properties
    {
        _CamIn ("Camera Input", 2D) = "black" {}
        _FrameBuffer ("Frame Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "SimpNetInclude.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            Texture2D<float3> _CamIn;
            Texture2D<float3> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                col = _Time.y < 1.0 ? 0..xxx : col;

                [branch]
                if (insideArea(txKern1Area, px))
                {
                    col.r = 1.0;
                }
                else if (insideArea(txBias1Area, px))
                {
                    col.r = 0.9;
                }
                else if (insideArea(txConv1Area, px))
                {
                    col.r = 0.8;
                }
                else if (insideArea(txMax1Area, px))
                {
                    col.r = 0.7;
                }
                else if (insideArea(txiMax1Area, px))
                {
                    col.r = 0.6;
                }
                else if (insideArea(txKern2Area, px))
                {
                    col.r = 0.5;
                }
                else if (insideArea(txBias2Area, px))
                {
                    col.r = 0.4;
                }
                else if (insideArea(txConv2Area, px))
                {
                    col.r = 0.3;
                }
                else if (insideArea(txMax2Area, px))
                {
                    col.r = 0.2;
                }
                else if (insideArea(txiMax2Area, px))
                {
                    col.r = 0.1;
                }
                else if (insideArea(txKern3Area, px))
                {
                    col.g = 1.0;
                }
                else if (insideArea(txBias3Area, px))
                {
                    col.g = 0.9;
                }
                else if (insideArea(txConv3Area, px))
                {
                    col.g = 0.8;
                }
                else if (insideArea(txMax3Area, px))
                {
                    col.g = 0.7;
                }
                else if (insideArea(txiMax3Area, px))
                {
                    col.g = 0.6;
                }
                else if (insideArea(txW1Area, px))
                {
                    col.g = 0.5;
                }
                else if (insideArea(txW1BiasArea, px))
                {
                    col.g = 0.4;
                }
                else if (insideArea(txFC1s, px))
                {
                    col.g = 0.3;
                }
                else if (insideArea(txFC1a, px))
                {
                    col.g = 0.2;
                }
                else if (insideArea(txW2Area, px))
                {
                    col.g = 0.1;
                }
                else if (insideArea(txW2BiasArea, px))
                {
                    col.b = 0.9;
                }
                else if (insideArea(txFC2s, px))
                {
                    col.b = 0.8;
                }
                else if (insideArea(txFC2a, px))
                {
                    col.b = 0.7;
                }
                else if (insideArea(txW3Area, px))
                {
                    col.b = 0.6;
                }
                else if (insideArea(txW3BiasArea, px))
                {
                    col.b = 0.5;
                }
                else if (insideArea(txSoftout1, px))
                {
                    col.b = 0.4;
                }
                else if (insideArea(txSoftout2, px))
                {
                    col.b = 0.3;
                }

                return col;
            }
            ENDCG
        }
    }
}