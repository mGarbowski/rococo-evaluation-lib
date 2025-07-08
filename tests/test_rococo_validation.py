from rococo_evaluation_lib import RococoValidation


def test_stores_parameter():
    validator = RococoValidation("test_param")
    assert validator.some_param == "test_param"
