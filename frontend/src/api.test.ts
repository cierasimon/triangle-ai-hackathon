import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("recognition upload mode", () => {
  it.each(["opencv", "hybrid"] as const)("sends the selected %s mode with the photo", async mode => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ image_id: "img_test" }), {
      status: 200, headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetch);
    const photo = new File(["fixture"], "diagram.jpg", { type: "image/jpeg" });
    await api.upload(photo, mode);
    const [path, request] = fetch.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/api/upload");
    expect(request.credentials).toBe("include");
    const form = request.body as FormData;
    expect(form.get("vision_mode")).toBe(mode);
    expect((form.get("photo") as File).name).toBe("diagram.jpg");
  });

  it("keeps older callers on local OpenCV by default", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    await api.upload(new File(["fixture"], "diagram.jpg"));
    expect((fetch.mock.calls[0][1].body as FormData).get("vision_mode")).toBe("opencv");
  });
});
