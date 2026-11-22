'use strict';
const video = document.querySelector('#demo');
const status = document.querySelector('#playback-status');
const sound = document.querySelector('#sound');
const fallback = document.querySelector('#play-fallback');
const chapters = [...document.querySelectorAll('[data-start]')];
video.muted = true;
async function play() {
  try { await video.play(); fallback.hidden = true; }
  catch { status.textContent = 'Your browser requires a tap to start.'; fallback.hidden = false; }
}
function updateStatus() {
  status.textContent = `${video.paused ? 'Paused' : 'Playing'} · ${video.muted ? 'Sound off' : 'Sound on'}`;
  sound.textContent = video.muted ? 'Enable sound' : 'Mute sound';
  sound.setAttribute('aria-pressed', String(!video.muted));
}
sound.addEventListener('click', () => { video.muted = !video.muted; updateStatus(); });
fallback.addEventListener('click', play);
for (const event of ['playing', 'pause', 'volumechange']) video.addEventListener(event, updateStatus);
chapters.forEach(button => button.addEventListener('click', () => { video.currentTime = Number(button.dataset.start); play(); }));
video.addEventListener('timeupdate', () => {
  const current = chapters.filter(button => Number(button.dataset.start) <= video.currentTime + 0.01).at(-1);
  chapters.forEach(button => button.setAttribute('aria-current', String(button === current)));
});
play();
