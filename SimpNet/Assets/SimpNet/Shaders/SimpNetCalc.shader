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
            Texture2D<float> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;

            float pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float col = LoadValue(_FrameBuffer, px);

                [branch]
                if (insideArea(txKern1Area, px))
                {
                    col = 1.0;
                }
                else if (insideArea(txBias1Area, px))
                {
                    col = 0.9;
                }
                else if (insideArea(txConv1Area, px))
                {
                    col = 0.8;
                }
                else if (insideArea(txMax1Area, px))
                {
                    col = 0.7;
                }
                else if (insideArea(txiMax1Area, px))
                {
                    col = 0.6;
                }

                return col;
            }
            ENDCG
        }
    }
}