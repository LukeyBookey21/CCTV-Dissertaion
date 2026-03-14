import subprocess
import pathlib


def merge(folder_path, output_path):
    folder = pathlib.Path(folder_path)
    videos = sorted(folder.rglob("*.mp4"), key=lambda p: str(p))
    if not videos:
        print(f"No videos found in {folder}")
        return
    print(f"Found {len(videos)} files")
    print(f"First: {videos[0]}")
    print(f"Last:  {videos[-1]}")

    concat_file = pathlib.Path("concat_list.txt")
    with open(concat_file, "w") as f:
        for v in videos:
            f.write(f"file '{v}'\n")

    print(f"\nMerging to {output_path} ...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            "copy",
            "-an",
            str(output_path),
        ],
        check=True,
    )
    print("Done!")
    concat_file.unlink()


if __name__ == "__main__":
    merge(
        r"C:\Users\lukew\OneDrive\Dokumente\garage_8-6",
        r"C:\Users\lukew\CCTV-Dissertaion\data\uploads\garage_merged.mp4",
    )
