# pyright: basic
# import the modules
import os
import time

from watchdog.events import DirCreatedEvent, FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from eval_utils import get_score

todo_evals: set[tuple[int, int]] = set()


class MyHandler(FileSystemEventHandler):
    def on_created(self, event: DirCreatedEvent | FileCreatedEvent):
        if isinstance(event, DirCreatedEvent):
            return
        try:
            if isinstance(event.src_path, bytes):
                event.src_path = event.src_path.decode()
            seed = event.src_path.split("/")[-1].split("_")[0]
            number = event.src_path.split("/")[-1].split("_")[1].split(".")[0]
            todo_evals.add((int(seed), int(number)))
        except KeyError:
            return


if __name__ == "__main__":

    # Set format for displaying path
    path = "output"
    event_handler = MyHandler()
    # Initialize Observer
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    # Start the observer
    observer.start()
    try:
        while True:
            # Get the score in current merged
            if todo_evals:
                seed, number = todo_evals.pop()
                print(f"seed: {seed}, number: {number}")
                current_score = get_score(f"merged/{seed}.json", seed)
                print(f"current score: {current_score}")
                new_score = get_score(f"output/{seed}_{number}.json", seed)
                if new_score > current_score:
                    print("new score is better:", new_score)
                    os.system(f"mv output/{seed}_{number}.json merged/{seed}.json")
                elif new_score < current_score:
                    print("new score is worse:", new_score)
                    os.system(f"rm output/{seed}_{number}.json")
                print("--------------------------------")
            else:
                time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
