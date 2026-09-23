import math
import unittest
import numpy as np
from drawer_metrics import quaternion_angle_deg, payload_in_initial_world, contact_force_from_impulse


class DrawerMetrics(unittest.TestCase):
    def test_equivalent_quaternion_signs(self):
        self.assertAlmostEqual(quaternion_angle_deg([0,0,0,1],[0,0,0,-1]),0)

    def test_quaternion_magnitude_does_not_hide_rotation(self):
        q=[0,0,2*math.sin(math.pi/4),2*math.cos(math.pi/4)]
        self.assertAlmostEqual(quaternion_angle_deg([0,0,0,2],q),90)

    def test_zero_quaternion_rejected(self):
        with self.assertRaises(ValueError):quaternion_angle_deg([0,0,0,0],[0,0,0,1])

    def test_translation_relative_frame(self):
        actual=payload_in_initial_world([.2,1.1,.4],[0,0,.3,0,0,0,1],[0,0,0,0,0,0,1])
        np.testing.assert_allclose(actual,[.2,1.1,.1],atol=1e-12)

    def test_rotated_and_translated_frame(self):
        q=[0,0,math.sin(math.pi/4),math.cos(math.pi/4)]
        actual=payload_in_initial_world([5,7,0],[5,5,0,*q],[0,0,0,0,0,0,1])
        np.testing.assert_allclose(actual,[2,0,0],atol=1e-12)

    def test_nonidentity_initial_frame(self):
        q=[0,0,math.sin(math.pi/4),math.cos(math.pi/4)]
        actual=payload_in_initial_world([5,7,0],[5,5,0,*q],[1,2,0,*q])
        np.testing.assert_allclose(actual,[1,4,0],atol=1e-12)

    def test_weight_impulse_force_units(self):
        impulse=np.array([0,.5*9.81/240,0])
        np.testing.assert_allclose(contact_force_from_impulse(impulse,1/240),[0,4.905,0],atol=1e-12)

    def test_nonpositive_dt_rejected(self):
        for dt in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):contact_force_from_impulse([0,1,0],dt)


if __name__=='__main__':unittest.main()
