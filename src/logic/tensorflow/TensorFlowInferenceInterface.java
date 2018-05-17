/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modified: simplified to fit the needs of the current project
==============================================================================*/

package logic.tensorflow;

import org.tensorflow.*;
import org.tensorflow.types.UInt8;

import java.io.IOException;
import java.nio.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class TensorFlowInferenceInterface {
    private final Graph g;
    private final Session sess;
    private Session.Runner runner;
    private List<String> feedNames = new ArrayList();
    private List<Tensor<?>> feedTensors = new ArrayList();
    private List<String> fetchNames = new ArrayList();
    private List<Tensor<?>> fetchTensors = new ArrayList();

    public TensorFlowInferenceInterface(Path path) {
        this.prepareNativeRuntime();
        this.g = new Graph();
        this.sess = new Session(this.g);
        this.runner = this.sess.runner();

        try {
            this.loadGraph(readAllBytesOrExit(path), this.g);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    public void run(String[] var1) {
        this.run(var1, false);
    }

    public void run(String[] var1, boolean var2) {
        this.closeFetches();
        String[] var3 = var1;
        int var4 = var1.length;

        for (int var5 = 0; var5 < var4; ++var5) {
            String var6 = var3[var5];
            this.fetchNames.add(var6);
            TensorFlowInferenceInterface.TensorId var7 = TensorFlowInferenceInterface.TensorId.parse(var6);
            this.runner.fetch(var7.name, var7.outputIndex);
        }

        try {
            if (var2) {
                Session.Run var13 = this.runner.runAndFetchMetadata();
                this.fetchTensors = var13.outputs;
            } else {
                this.fetchTensors = this.runner.run();
            }
        } catch (RuntimeException var11) {
            System.out.println("Failed to run TensorFlow inference with inputs:["
                    + String.join(", ", this.feedNames)
                    + "], outputs:[" + String.join(", ", this.fetchNames) + "]");
            throw var11;
        } finally {
            this.closeFeeds();
            this.runner = this.sess.runner();
        }

    }

    public Graph graph() {
        return this.g;
    }

    public Operation graphOperation(String var1) {
        Operation var2 = this.g.operation(var1);
        if (var2 == null) {
            throw new RuntimeException("Node '" + var1 + "' does not exist in model '");
        } else {
            return var2;
        }
    }

    public void close() {
        this.closeFeeds();
        this.closeFetches();
        this.sess.close();
        this.g.close();
    }

    protected void finalize() throws Throwable {
        try {
            this.close();
        } finally {
            super.finalize();
        }

    }

    public void feed(String var1, float[] var2, long... var3) {
        this.addFeed(var1, Tensor.create(var3, FloatBuffer.wrap(var2)));
    }

    public void feed(String var1, int[] var2, long... var3) {
        this.addFeed(var1, Tensor.create(var3, IntBuffer.wrap(var2)));
    }

    public void feed(String var1, long[] var2, long... var3) {
        this.addFeed(var1, Tensor.create(var3, LongBuffer.wrap(var2)));
    }

    public void feed(String var1, double[] var2, long... var3) {
        this.addFeed(var1, Tensor.create(var3, DoubleBuffer.wrap(var2)));
    }

    public void feed(String var1, byte[] var2, long... var3) {
        this.addFeed(var1, Tensor.create(UInt8.class, var3, ByteBuffer.wrap(var2)));
    }

    public void feedString(String var1, byte[] var2) {
        this.addFeed(var1, Tensors.create(var2));
    }

    public void feedString(String var1, byte[][] var2) {
        this.addFeed(var1, Tensors.create(var2));
    }

    public void feed(String var1, FloatBuffer var2, long... var3) {
        this.addFeed(var1, Tensor.create(var3, var2));
    }

    public void feed(String var1, IntBuffer var2, long... var3) {
        this.addFeed(var1, Tensor.create(var3, var2));
    }

    public void feed(String var1, LongBuffer var2, long... var3) {
        this.addFeed(var1, Tensor.create(var3, var2));
    }

    public void feed(String var1, DoubleBuffer var2, long... var3) {
        this.addFeed(var1, Tensor.create(var3, var2));
    }

    public void feed(String var1, ByteBuffer var2, long... var3) {
        this.addFeed(var1, Tensor.create(UInt8.class, var3, var2));
    }

    public void fetch(String var1, float[] var2) {
        this.fetch(var1, FloatBuffer.wrap(var2));
    }

    public void fetch(String var1, int[] var2) {
        this.fetch(var1, IntBuffer.wrap(var2));
    }

    public void fetch(String var1, long[] var2) {
        this.fetch(var1, LongBuffer.wrap(var2));
    }

    public void fetch(String var1, double[] var2) {
        this.fetch(var1, DoubleBuffer.wrap(var2));
    }

    public void fetch(String var1, byte[] var2) {
        this.fetch(var1, ByteBuffer.wrap(var2));
    }

    public void fetch(String var1, FloatBuffer var2) {
        this.getTensor(var1).writeTo(var2);
    }

    public void fetch(String var1, IntBuffer var2) {
        this.getTensor(var1).writeTo(var2);
    }

    public void fetch(String var1, LongBuffer var2) {
        this.getTensor(var1).writeTo(var2);
    }

    public void fetch(String var1, DoubleBuffer var2) {
        this.getTensor(var1).writeTo(var2);
    }

    public void fetch(String var1, ByteBuffer var2) {
        this.getTensor(var1).writeTo(var2);
    }

    private void prepareNativeRuntime() {
        System.out.println("Checking to see if TensorFlow native methods are already loaded");

        try {
            System.out.println("TensorFlow native methods already loaded");
        } catch (UnsatisfiedLinkError var4) {
            System.out.println("TensorFlow native methods not found, attempting to load via tensorflow_inference");
        }

    }

    private void loadGraph(byte[] var1, Graph var2) throws IOException {
        long var3 = System.currentTimeMillis();
        try {
            var2.importGraphDef(var1);
        } catch (IllegalArgumentException var7) {
            throw new IOException("Not a valid TensorFlow Graph serialization: " + var7.getMessage());
        }


        long var5 = System.currentTimeMillis();
        System.out.println("Model load took " + (var5 - var3) + "ms, TensorFlow version: " + TensorFlow.version());
    }

    private void addFeed(String var1, Tensor<?> var2) {
        TensorFlowInferenceInterface.TensorId var3 = TensorFlowInferenceInterface.TensorId.parse(var1);
        this.runner.feed(var3.name, var3.outputIndex, var2);
        this.feedNames.add(var1);
        this.feedTensors.add(var2);
    }

    private Tensor<?> getTensor(String var1) {
        int var2 = 0;

        for (Iterator var3 = this.fetchNames.iterator(); var3.hasNext(); ++var2) {
            String var4 = (String) var3.next();
            if (var4.equals(var1)) {
                return (Tensor) this.fetchTensors.get(var2);
            }
        }

        throw new RuntimeException("Node '" + var1 + "' was not provided to run(), so it cannot be read");
    }

    private void closeFeeds() {
        Iterator var1 = this.feedTensors.iterator();

        while (var1.hasNext()) {
            Tensor var2 = (Tensor) var1.next();
            var2.close();
        }

        this.feedTensors.clear();
        this.feedNames.clear();
    }

    private void closeFetches() {
        Iterator var1 = this.fetchTensors.iterator();

        while (var1.hasNext()) {
            Tensor var2 = (Tensor) var1.next();
            var2.close();
        }

        this.fetchTensors.clear();
        this.fetchNames.clear();
    }

    private static class TensorId {
        String name;
        int outputIndex;

        private TensorId() {
        }

        public static TensorFlowInferenceInterface.TensorId parse(String var0) {
            TensorFlowInferenceInterface.TensorId var1 = new TensorFlowInferenceInterface.TensorId();
            int var2 = var0.lastIndexOf(58);
            if (var2 < 0) {
                var1.outputIndex = 0;
                var1.name = var0;
                return var1;
            } else {
                try {
                    var1.outputIndex = Integer.parseInt(var0.substring(var2 + 1));
                    var1.name = var0.substring(0, var2);
                } catch (NumberFormatException var4) {
                    var1.outputIndex = 0;
                    var1.name = var0;
                }

                return var1;
            }
        }
    }

}
