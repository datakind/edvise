from edvise.dataio.batch_frame_matching import bind_filenames_to_datasets


def test_bind_filenames_prefers_higher_score_and_does_not_reuse_files() -> None:
    bound = bind_filenames_to_datasets(
        ["student_2024.csv", "course_2024.csv", "notes.csv"],
        {
            "student": ["raw_student.csv"],
            "course": ["raw_course.csv"],
        },
    )
    assert bound == {
        "student": "student_2024.csv",
        "course": "course_2024.csv",
    }
