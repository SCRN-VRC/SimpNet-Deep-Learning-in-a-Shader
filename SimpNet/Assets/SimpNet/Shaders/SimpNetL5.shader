Shader "SimpNet/SimpNetL5"
{
    Properties
    {
        _Layer1 ("Layer 4", 2D) = "black" {}
        _FrameBuffer ("Layer 5 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 5"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "SimpNetLayout.cginc"
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
                if (insideArea(txW2Area, px))
                {
                    col.r = 1.0;
                }
                else if (insideArea(txW2BiasArea, px))
                {
                    col.r = 0.9;
                }
                else if (insideArea(txFC2s, px))
                {
                    col.r = 0.8;
                }
                else if (insideArea(txFC2a, px))
                {
                    col.r = 0.7;
                }

                return col;
            }
            ENDCG
        }
    }
}