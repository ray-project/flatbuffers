// automatically generated, do not modify

namespace MyGame.Example
{

using System;
using FlatBuffers;

public sealed class Test : Struct {
  public Test __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; return this; }

  public short A { get { return bb.GetShort(bb_pos + 0); } }
  public void MutateA(short a) { bb.PutShort(bb_pos + 0, a); }
  public sbyte B { get { return bb.GetSbyte(bb_pos + 2); } }
  public void MutateB(sbyte b) { bb.PutSbyte(bb_pos + 2, b); }

  public static Offset<Test> CreateTest(FlatBufferBuilder builder, short A, sbyte B) {
    builder.Prep(2, 4);
    builder.Pad(1);
    builder.PutSbyte(B);
    builder.PutShort(A);
    return new Offset<Test>(builder.Offset);
  }
};


}
