# Photographs

Drop football photos here and the "My season" tab shows them. No code changes.

```bash
python tools/optimise_photos.py ~/Pictures/some-folder
python tools/optimise_photos.py shot1.jpg shot2.jpg shot3.jpg
```

## Rules the tool enforces for you

- **120 KB per image, hard.** These are drawn on phones, often on a metered
  bundle. `st.image` serves each one from Streamlit's media endpoint so it is
  cached after the first view, but the first view is still the one that
  matters. A photo that will not compress that far is dropped with a message
  rather than quietly shipped.
- **Four images maximum.** That is what the strip displays. More is pure weight.
- **Resized into an 800x560 box**, which is roughly 3x the size they are drawn
  at. EXIF orientation is honoured, so portrait shots do not end up sideways.

## Captions

The caption is the filename, so renaming the file is how you edit it.

`01_katowice-internationals-2024.jpg` renders as "katowice internationals 2024".

The leading `NN_` sets the order and is not shown.

## What is here now

`01_katowice-internationals-2024.jpg` is a placeholder: it is Percival at the
Katowice Internationals in 2024, standing in front of the group results board,
which suits a page about predicting results. **It is not him playing.** The
green Zimbabwe kit photos he wanted are not on this machine, only pasted into a
chat. Drop them in this folder, run the command above, and delete this one.
